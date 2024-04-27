import os
import time

import openmldb.dbapi
import logging
import pandas as pd
from numpy import dtype
from utils import get_create_table_sql_in_workspace, get_window_sql_in_workspace, get_train_df_csv_in_workspace

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s %(name)-12s %(levelname)-4s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__file__)

db = None
cursor = None
online_mode = False
workspace_path = ""
db_name = "db1"
table_name = "automl"
proc_name = "window_deploy"
window_sql_parts = []


def init(workspace, online=False):
    logger.info(f"init, online_mode:{online}")
    global db, cursor, online_mode, workspace_path
    online_mode, workspace_path = online, workspace
    retry = 60
    while retry > 0:
        try:
            db = openmldb.dbapi.connect(zk="0.0.0.0:2181", zkPath="/openmldb")
            break
        except Exception as e:
            logger.warning(e)
            logger.info(f"left retry: {retry}")
            time.sleep(5)
            retry -= 1
    cursor = db.cursor()
    try:
        cursor.execute(f"CREATE DATABASE {db_name}")
    except openmldb.dbapi.dbapi.DatabaseError as e:
        logger.warning(e)
    cursor.execute(f"USE {db_name}")
    execute_mode = "'online'" if online else "'offline'"
    cursor.execute(f"set @@execute_mode={execute_mode};")

    if online_mode:
        try:
            create_table_sql_path = get_create_table_sql_in_workspace(workspace_path)
            logger.info(f"Load CreateTableSql From {create_table_sql_path}")
            with open(create_table_sql_path, "r") as fp:
                create_table_sql = fp.read()
            logger.info(f"CreateTableSql: {create_table_sql}")
            # create_table_sql = create_table_sql[:-1] + ", INDEX(KEY=(reqId, userId), TS=eventTime))"
            # logger.info(f"new create: {create_table_sql}")
            try:
                cursor.execute(create_table_sql)
            except openmldb.dbapi.dbapi.DatabaseError as e:
                logger.warning(e)

            train_df_csv_path = get_train_df_csv_in_workspace(workspace_path)
            load_data_infile(table_name, train_df_csv_path)
            logger.info("Finished loading train data")

            window_sql_path = get_window_sql_in_workspace(workspace_path)
            logger.info(f"Load WindowSql From {window_sql_path}")
            global window_sql_parts
            with open(window_sql_path, "r") as fp:
                window_sql = fp.read()
            logger.info(f"WindowSql: {window_sql}")
            window_sql_parts = window_sql.split('WINDOW')
            logger.info(f"window_sql_parts: {window_sql_parts}")
        except Exception as ex:
            logger.warning(ex)
    logger.info(f"Finished init")


def append_window_union_features(df, number_cols, id_col, partition_by_col, sort_by_col):
    logger.info("Start to write df to openmldb")
    write_df_to_openmldb(df, table_name)
    logger.info("Finished to write df to openmldb")
    return window(df, table_name, number_cols, id_col, partition_by_col, sort_by_col)


def write_df_to_openmldb(df, table):
    dtypes = df.dtypes.to_dict()
    col_infos = []
    timestamp_cols = []
    for col_name, raw_type in dtypes.items():
        if raw_type == dtype('int32'):
            col_type = 'int'
        elif raw_type == dtype('int64'):
            col_type = 'bigint'
        elif raw_type == dtype('float64'):
            col_type = 'double'
        elif raw_type == dtype('bool'):
            col_type = 'bool'
        elif raw_type == dtype('<M8[ns]'):
            col_type = 'timestamp'
            timestamp_cols.append(col_name)
        else:
            col_type = 'string'
        col_infos.append(col_name + " " + col_type)
    df[timestamp_cols] = (df[timestamp_cols].astype(dtype('int64')) / 1000000).astype(dtype('int64'))

    if not online_mode:
        try:
            cursor.execute(f"DROP TABLE {table}")
        except openmldb.dbapi.dbapi.DatabaseError as e:
            logger.warning(e)

        sql = f"CREATE TABLE {table} ({', '.join(col_infos)})"
        cursor.execute(sql)
        create_table_sql_path = get_create_table_sql_in_workspace(workspace_path)
        with open(create_table_sql_path, "w") as fp:
            fp.write(sql)
        logger.info(f"Write CreateTableSql to {create_table_sql_path}")

    write_to_db(df, table)


def write_to_db(df, table):
    start_time = time.time()
    if online_mode:
        cols = ', '.join(df.columns)
        df.apply(lambda x: insert(x, table, cols), axis=1)
    else:
        train_df_csv_path = get_train_df_csv_in_workspace(workspace_path)
        df.to_csv(train_df_csv_path, index=False, header=True, encoding="utf-8")
        load_data_infile(table, train_df_csv_path)

    end_time = time.time()
    print("start time: " + str(start_time))
    print("end time: " + str(end_time))
    print("cost: " + str(end_time - start_time))
    logger.info("Data written to openMLDB successfully!")


def insert(row, table, cols):
    values_str = ', '.join([f"'{value}'" if isinstance(value, str) else str(value) for value in row])
    insert_sql = f"INSERT INTO {table} ({cols}) VALUES ({values_str});"
    cursor.execute(insert_sql)


def load_data_infile(table: str, file_path: str, mode: str = 'overwrite', block: bool = True):
    file_path = f"file://{os.path.abspath(file_path)}"
    deep_copy = "true" if online_mode else "false"
    if online_mode:
        mode = "append"
    cursor.execute(
        f"LOAD DATA INFILE '{file_path}' INTO TABLE {table} options(format='csv', mode='{mode}', deep_copy={deep_copy});")
    job_id = None
    for job_info in cursor.fetchall():
        job_id = job_info[0]
    if block:
        job_state = get_job_state(job_id)
        while job_state.lower() not in ['failed', 'finished', 'killed']:
            logger.info(f'load data infile job {job_id} state: {job_state}')
            time.sleep(5)
            job_state = get_job_state(job_id)
        logger.info(f'load data infile job {job_id} state: {job_state}')
    return job_id


def get_job_state(job_id: str) -> str:
    cursor.execute(f"show job {job_id}")
    for job_info in cursor.fetchall():
        return job_info[2]


def window(df, table, cols, id_col, partition_by_col, order_by_col):
    agg_cols = []
    agg_col_sqls = []
    agg_funcs = ["max", "min", "avg"]
    for agg_func in agg_funcs:
        for col_name in cols:
            agg_cols.append(f"{col_name}_{agg_func}")
            agg_col_sqls.append(f"{agg_func}({col_name}) OVER w AS {col_name}_{agg_func}")

    if online_mode:
        print(agg_cols)
        start_time = time.time()
        # df[agg_cols] = df.apply(lambda x: row_call_proc(x, id_col), axis=1)
        # reqId需保证唯一
        ids = ', '.join([f"'{reqId}'" for reqId in df[id_col]])
        sql = f"{window_sql_parts[0]}WHERE {id_col} in ({ids}) WINDOW{window_sql_parts[1]}"
        result = cursor.execute(sql)
        res_tuple = result.fetchall()
        all_cols = agg_cols.copy()
        all_cols.insert(0, id_col)
        df_aggs = pd.DataFrame(res_tuple, columns=all_cols)
        logger.info(len(df_aggs))
        df = pd.merge(df, df_aggs, on=id_col, how="left")
        end_time = time.time()
        logger.info("get window union features cost time: " + str(end_time - start_time))
    else:
        agg_col_sql_str = ", ".join(agg_col_sqls)
        sql = f"SELECT {id_col}, {agg_col_sql_str} FROM {table} " \
              f"WINDOW w AS (PARTITION BY {partition_by_col} ORDER BY {order_by_col} " \
              f"ROWS BETWEEN 50 PRECEDING AND CURRENT ROW)"
        logger.info("sql: " + sql)

        window_sql_path = get_window_sql_in_workspace(workspace_path)
        with open(window_sql_path, "w") as fp:
            fp.write(sql)

        all_cols = agg_cols.copy()
        all_cols.insert(0, id_col)
        df_agg_cols = get_window_df(sql, all_cols)

        df = pd.merge(df, df_agg_cols, on=id_col, how="left")
    df[agg_cols] = df[agg_cols].fillna(0)
    return df, agg_cols


def row_call_proc(row, id_col):
    sql = f"{window_sql_parts[0]}WHERE {id_col}='{row[id_col]}' WINDOW{window_sql_parts[1]}"
    result = cursor.execute(sql)
    res_tuple = result.fetchone()
    return pd.Series(res_tuple[1:])


def get_window_df(sql, cols):
    export_csv_dir = "window_union"
    export_new_feature_outfile(sql, export_csv_dir)
    csv_files = os.listdir(export_csv_dir)
    csv_files.sort()
    df_parts = []
    for file in csv_files:
        if not file.endswith(".csv"):
            continue
        file_path = os.path.join(export_csv_dir, file)
        csv_f = pd.read_csv(file_path)
        df = pd.DataFrame(csv_f, columns=cols)
        df_parts.append(df)
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df = df.reset_index(drop=True)
    else:
        df = df_parts[0]
    return df


def export_new_feature_outfile(sql: str, file_path: str, mode: str = 'overwrite',
                               block: bool = True):
    file_path = f"file://{os.path.abspath(file_path)}"
    cursor.execute(
        f"{sql} INTO OUTFILE '{file_path}' options(delimiter=',', format='csv', mode='{mode}')")
    job_id = None
    for job_info in cursor.fetchall():
        job_id = job_info[0]
    if block:
        job_state = get_job_state(job_id)
        while job_state.lower() not in ['failed', 'finished', 'killed']:
            logger.info(f'export new feature outfile job {job_id} state: {job_state}')
            time.sleep(5)
            job_state = get_job_state(job_id)
        logger.info(f'export new feature outfile job {job_id} state: {job_state}')
    return job_id


def test():
    data = {'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'score': [1.1, 2.2, 3.3],
            'ts': [pd.Timestamp.utcnow().timestamp(), pd.Timestamp.utcnow().timestamp(),
                   pd.Timestamp.utcnow().timestamp()],
            'dt': [pd.to_datetime('20240101', format='%Y%m%d'), pd.to_datetime('20240201', format='%Y%m%d'),
                   pd.to_datetime('20240301', format='%Y%m%d')],
            'c1': [True, True, False],
            'c2': [pd.to_datetime('2023-12-14 00:03:05.662000'), pd.to_datetime('2023-12-14 00:03:05.662000'),
                   pd.to_datetime('2023-12-14 00:03:05.662000')]
            }
    df = pd.DataFrame(data)
    write_df_to_openmldb(df, 't4')


if __name__ == "__main__":
    test()
