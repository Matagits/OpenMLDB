import glob
import os
import time

import openmldb.dbapi
import logging
import pandas as pd
from numpy import dtype

from autofe import OpenMLDBSQLGenerator, AutoXTrain
from utils import get_create_table_sql_in_workspace, get_index_sql_in_workspace, get_window_sql_in_workspace, get_train_df_csv_in_workspace
from sklearn.model_selection import train_test_split

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
window_sql = ""


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
    cursor.execute(f"set @@spark_config='spark.driver.memory=4g';")

    if online_mode:
        try:
            create_table_sql_path = get_create_table_sql_in_workspace(workspace_path)
            logger.info(f"Load CreateTableSql From {create_table_sql_path}")
            with open(create_table_sql_path, "r") as fp:
                create_table_sql = fp.read()
            logger.info(f"CreateTableSql: {create_table_sql}")
            index_sql_path = get_index_sql_in_workspace(workspace_path)
            logger.info(f"Load IndexSql From {index_sql_path}")
            with open(index_sql_path, "r") as fp:
                index_sql = fp.read()
            create_table_sql = f"{create_table_sql[:-1]}, {index_sql})"
            logger.info(f"new create: {create_table_sql}")
            try:
                cursor.execute(create_table_sql)
            except openmldb.dbapi.dbapi.DatabaseError as e:
                logger.warning(e)

            train_df_csv_path = get_train_df_csv_in_workspace(workspace_path)
            load_data_infile(table_name, train_df_csv_path)
            logger.info("Finished loading train data")

            window_sql_path = get_window_sql_in_workspace(workspace_path)
            logger.info(f"Load WindowSql From {window_sql_path}")
            global window_sql
            with open(window_sql_path, "r") as fp:
                window_sql = fp.read()
            logger.info(f"WindowSql: {window_sql}")
        except Exception as ex:
            logger.warning(ex)
    logger.info(f"Finished init")


def append_window_union_features(df, label, number_cols, id_col, partition_by_col, sort_by_col):
    # print(df.head())
    # print(df[[id_col, partition_by_col, sort_by_col]].head())
    print(label.head())

    logger.info("Start to write df to openmldb")
    write_df_to_openmldb(df, table_name)
    logger.info("Finished to write df to openmldb")
    return window(df, label, table_name, number_cols, id_col, partition_by_col, sort_by_col)


def write_df_to_openmldb(df_origin, table):
    df = df_origin.copy()
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
    logger.info("write_to_db cost time: " + str(end_time - start_time))
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


def window(df, label, table, cols, id_col, partition_by_col, order_by_col):
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
        # df[agg_cols] = df.apply(row_call_proc, axis=1)
        temp = df.apply(lambda x: tuple(x), axis=1)
        temp_list = temp.values.tolist()
        result = cursor.execute(f"{window_sql} CONFIG (execute_mode = 'request', values = {temp_list})")
        res_tuple = result.fetchall()
        result_df = pd.DataFrame(res_tuple)
        all_cols = agg_cols.copy()
        all_cols.insert(0, id_col)
        result_df.columns = all_cols
        df = pd.merge(df, result_df, on=id_col, how="left")
        end_time = time.time()
        logger.info("get window union features cost time: " + str(end_time - start_time))
    else:

        logger.info("autofe")
        conf = {
            'tables': [
                {
                    'table': table_name
                }
            ],
            'main_table': table_name,
            'windows': [
                {
                    'name': 'w1',
                    'partition_by': partition_by_col,
                    'order_by': order_by_col,
                    'window_type': 'rows_range',
                    'start': '1d PRECEDING',
                    'end': 'CURRENT ROW'
                }
            ]
        }
        sql_generator = OpenMLDBSQLGenerator(conf, df)
        sql, feature_path = sql_generator.time_series_feature_sql()
        logger.error(f"time_series_feature_sql: {sql}")

        # agg_col_sql_str = ", ".join(agg_col_sqls)
        # sql = f"SELECT {id_col}, {agg_col_sql_str} FROM {table} " \
        #       f"WINDOW w AS (PARTITION BY {partition_by_col} ORDER BY {order_by_col} " \
        #       f"ROWS BETWEEN 50 PRECEDING AND CURRENT ROW)"
        logger.info("sql: " + sql)

        window_sql_path = get_window_sql_in_workspace(workspace_path)
        with open(window_sql_path, "w") as fp:
            fp.write(sql)

        index_sql_path = get_index_sql_in_workspace(workspace_path)
        with open(index_sql_path, "w") as fp:
            fp.write(f"index(key={partition_by_col}, ttl=50, ttl_type=latest, ts=`{order_by_col}`)")

        all_cols = agg_cols.copy()
        all_cols.insert(0, id_col)
        df_agg_cols = get_window_df(sql, all_cols, feature_path)

        logger.error(f"df length: {len(df)}")
        logger.error(f"df_agg_cols length: {len(df_agg_cols)}")
        df = pd.merge(df, df_agg_cols, on=id_col, how="left")

        label.index = df.index
        logger.error(f"df length: {len(df)}")
        logger.error(f"label length: {len(label)}")
        logger.error(f"df columns: {df.columns.tolist()}")
        df['automl_label'] = label
        logger.error(f"after df columns: {df.columns.tolist()}")

        df.drop(columns=['eventTime'], inplace=True)
        train_set, test_set_with_y = train_test_split(df, train_size=0.8)
        test_set = test_set_with_y.drop(columns=['automl_label'])

        offline_feature_path = '/tmp/automl_offline_feature'

        # save for backup
        train_name = 'train.parquet'
        test_name = 'test.parquet'
        train_set.to_parquet(offline_feature_path + '/' + train_name, index=False)
        test_set.to_parquet(offline_feature_path + '/' + test_name, index=False)

        id_list = [id_col]
        topk = 10
        logger.error(f'get top {topk} features')
        topk_features = AutoXTrain(debug=True).get_top_features(
            train_set, test_set, id_list, 'automl_label', offline_feature_path, topk)
        logger.error(f'top {len(topk_features)} feas: {topk_features}')

        # decode feature to final sql
        final_sql = sql_generator.decode_time_series_feature_sql_column(
            topk_features)
        logger.error(f'final sql: {final_sql}')

    df[agg_cols] = df[agg_cols].fillna(0)
    return df, agg_cols


def row_call_proc(row):
    value_tuple = tuple(row)
    # logger.info(f"{window_sql} CONFIG (execute_mode = 'request', values = {value_tuple})")
    result = cursor.execute(f"{window_sql} CONFIG (execute_mode = 'request', values = {value_tuple})")
    res_tuple = result.fetchone()
    return pd.Series(res_tuple[1:])


def get_window_df(sql, cols, feature_path):
    export_csv_dir = "window_union"
    export_new_feature_outfile(sql, export_csv_dir)

    def remove_prefix(text, prefix): return text[len(
        prefix):] if text.startswith(prefix) else text
    feature_path = remove_prefix(feature_path, 'file://')
    logging.info(f'load {feature_path}')
    df = pd.concat(map(pd.read_parquet, glob.glob(
        os.path.join('', feature_path + '/*.parquet'))))

    # csv_files = os.listdir(feature_path)
    # csv_files.sort()
    # df_parts = []
    # for file in csv_files:
    #     if not file.endswith(".parquet"):
    #         continue
    #     file_path = os.path.join(export_csv_dir, file)
    #     csv_f = pd.read_parquet(file_path)
    #     df = pd.DataFrame(csv_f)
    #     df_parts.append(df)
    # if len(df_parts) > 0:
    #     df = pd.concat(df_parts)
    #     df = df.reset_index(drop=True)
    # else:
    #     df = df_parts[0]
    return df


def export_new_feature_outfile(sql: str, file_path: str, mode: str = 'overwrite',
                               block: bool = True):
    file_path = f"file://{os.path.abspath(file_path)}"
    cursor.execute(sql)
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
