import openmldb.dbapi
import pandas as pd
from numpy import dtype
from logger import logger

db = None
cursor = None


def init():
    logger.error()
    global db, cursor
    db = openmldb.dbapi.connect(zk="0.0.0.0:2181", zkPath="/openmldb")
    cursor = db.cursor()
    # cursor.execute("CREATE DATABASE db1")
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

    sql = f"CREATE TABLE {table_name} ({', '.join(col_infos)})"
    cursor.execute(sql)

    for index, row in df.iterrows():
        values_str = ', '.join([f"'{value}'" if isinstance(value, str) else str(value) for value in row])
        insert_sql = f"INSERT INTO {table_name} ({', '.join(df.columns)}) VALUES ({values_str});"
        cursor.execute(insert_sql)


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
