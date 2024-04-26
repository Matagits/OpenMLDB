import os
import time

import openmldb.dbapi
import pandas as pd
from numpy import dtype
from logger import logger

db = None
cursor = None
online_mode = False


def init(online=False):
    logger.error("init")
    global db, cursor, online_mode
    online_mode = online
    retry = 60
    while retry > 0:
        try:
            db = openmldb.dbapi.connect(zk="0.0.0.0:2181", zkPath="/openmldb")
            break
        except Exception as e:
            logger.warn(e)
            logger.warn(retry)
            time.sleep(5)
            retry -= 1
    cursor = db.cursor()
    try:
        cursor.execute("CREATE DATABASE db1")
    except openmldb.dbapi.dbapi.DatabaseError as e:
        logger.warn(e)
    cursor.execute("USE db1")
    execute_mode = "'online'" if online else "'offline'"
    cursor.execute(f"set @@execute_mode={execute_mode};")


def write(df, table_name):
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

    try:
        cursor.execute(f"DROP TABLE {table_name}")
    except openmldb.dbapi.dbapi.DatabaseError as e:
        logger.warn(e)
    sql = f"CREATE TABLE {table_name} ({', '.join(col_infos)})"
    cursor.execute(sql)

    write_to_db(df, table_name)


def write_to_db(df, table_name):
    start_time = time.time()
    if online_mode:
        # for index, row in df.iterrows():
        #     values_str = ', '.join([f"'{value}'" if isinstance(value, str) else str(value) for value in row])
        #     insert_sql = f"INSERT INTO {table_name} ({', '.join(df.columns)}) VALUES ({values_str});"
        #     cursor.execute(insert_sql)

        cols = ', '.join(df.columns)
        df.apply(lambda row1: insert(row1, table_name, cols), axis=1)
    else:
        df.to_csv('raw_df.csv', index=False, header=True, encoding="utf-8")
        load_data_infile(table_name, "raw_df.csv")

    end_time = time.time()
    print("start time: " + str(start_time))
    print("end time: " + str(end_time))
    print("cost: " + str(end_time - start_time))
    print("Data written to openMLDB successfully!")


def insert(row, table_name, cols):
    values_str = ', '.join([f"'{value}'" if isinstance(value, str) else str(value) for value in row])
    insert_sql = f"INSERT INTO {table_name} ({cols}) VALUES ({values_str});"
    cursor.execute(insert_sql)


def load_data_infile(table_name: str, file_path: str, mode: str = 'overwrite', block: bool = True):
    file_path = f"file://{os.path.abspath(file_path)}"
    deep_copy = "true" if online_mode else "false"
    cursor.execute(
        f"LOAD DATA INFILE '{file_path}' INTO TABLE {table_name} options(format='csv', mode='{mode}', deep_copy={deep_copy});")
    job_id = None
    for job_info in cursor.fetchall():
        job_id = job_info[0]
    if block:
        job_state = get_job_state(job_id)
        while job_state.lower() not in ['failed', 'finished', 'killed']:
            print(f'load data infile job {job_id} state: {job_state}')
            time.sleep(5)
            job_state = get_job_state(job_id)
        print(f'load data infile job {job_id} state: {job_state}')
    return job_id


def get_job_state(job_id: str) -> str:
    cursor.execute(f"show job {job_id}")
    for job_info in cursor.fetchall():
        return job_info[2]


def window(table_name, cols, partition_by_col, order_by_col):
    agg_cols = []
    agg_col_sqls = []
    agg_funcs = ["max", "min", "avg"]
    for agg_func in agg_funcs:
        for col_name in cols:
            agg_cols.append(f"{col_name}_{agg_func}")
            agg_col_sqls.append(f"{agg_func}({col_name}) OVER w AS {col_name}_{agg_func}")
    agg_col_sql_str = ", ".join(agg_col_sqls)
    sql = f"SELECT reqId, {agg_col_sql_str} FROM {table_name} " \
          f"WINDOW w AS (PARTITION BY {partition_by_col} ORDER BY {order_by_col} " \
          f"ROWS BETWEEN 50 PRECEDING AND CURRENT ROW)"
    print("sql: " + sql)

    all_cols = agg_cols.copy()
    all_cols.insert(0, "reqId")
    df = get_window_df(sql, all_cols)
    return df, agg_cols


def get_window_df(sql, cols):
    if online_mode:
        result = cursor.execute(sql + ";")
        rows = result.fetchall()
        df = pd.DataFrame(rows, columns=cols)
    else:
        export_new_feature_outfile(sql, "window_union")

        csv_files = os.listdir("window_union")
        csv_files.sort()
        df_parts = []
        for file in csv_files:
            if not file.endswith(".csv"):
                continue
            file_path = os.path.join("window_union", file)
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
            print(f'export new feature outfile job {job_id} state: {job_state}')
            time.sleep(5)
            job_state = get_job_state(job_id)
        print(f'export new feature outfile job {job_id} state: {job_state}')
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
            'c2': [pd.to_datetime('2023-12-14 00:03:05.662000'),pd.to_datetime('2023-12-14 00:03:05.662000'),pd.to_datetime('2023-12-14 00:03:05.662000')]
            }
    df = pd.DataFrame(data)
    write(df, 't4')


if __name__ == "__main__":
    init()
    test()
