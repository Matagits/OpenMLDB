import openmldb.dbapi
import pandas as pd
from numpy import dtype
from logger import logger

db = None
cursor = None


def init():
    logger.error("init")
    global db, cursor
    db = openmldb.dbapi.connect(zk="0.0.0.0:2181", zkPath="/openmldb")
    cursor = db.cursor()
    try:
        cursor.execute("CREATE DATABASE db1")
    except openmldb.dbapi.dbapi.DatabaseError as e:
        logger.warn(e)
    cursor.execute("USE db1")


def read():
    pass


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

    for index, row in df.iterrows():
        values_str = ', '.join([f"'{value}'" if isinstance(value, str) else str(value) for value in row])
        insert_sql = f"INSERT INTO {table_name} ({', '.join(df.columns)}) VALUES ({values_str});"
        cursor.execute(insert_sql)


def window(table_name, cols, partition_by_col, order_by_col):
    agg_cols = []
    agg_col_sqls = []
    agg_funcs = ["max", "min", "avg"]
    for agg_func in agg_funcs:
        for col_name in cols:
            agg_cols.append(f"{col_name}_{agg_func}")
            agg_col_sqls.append(f"{agg_func}({col_name}) OVER w AS {col_name}_{agg_func}")
    agg_col_sql_str = ", ".join(agg_col_sqls)
    sql = f"SELECT {agg_col_sql_str} FROM {table_name} " \
          f"WINDOW w AS (PARTITION BY {partition_by_col} ORDER BY {order_by_col} " \
          f"ROWS BETWEEN 50 PRECEDING AND CURRENT ROW)"
    select_into_sql = f"{sql} INTO OUTFILE 'data.csv' OPTIONS (delimiter=',', mode='overwrite');"
    print("window sql: " + select_into_sql)
    cursor.execute(select_into_sql)
    result_csv = pd.read_csv('data.csv')
    df = pd.DataFrame(result_csv, columns=agg_cols)
    print(df.head())
    return df, agg_cols



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
