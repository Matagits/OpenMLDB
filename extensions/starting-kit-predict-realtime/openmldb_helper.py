import openmldb.dbapi
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy import Column, Date, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base


db = None
engine = None


def init():
    # global db
    # db = openmldb.dbapi.connect(zk="0.0.0.0:2181", zkPath="/openmldb")
    global engine
    engine = create_engine('mysql+pymysql://root:root@127.0.0.1:3307/db1', echo=True)


def read():
    query = "SELECT * FROM t2"
    with engine.connect() as conn:
        df = pd.read_sql(query, conn.connection)
    print(df.head())


def write(df, table_name):
    insert_sql_list = []
    for index, row in df.iterrows():
        values_str = ', '.join([f"'{value}'" if isinstance(value, str) else str(value) for value in row])
        insert_sql = f"INSERT INTO {table_name} ({', '.join(df.columns)}) VALUES ({values_str});"
        insert_sql_list.append(insert_sql)
    with engine.connect() as conn:
        for sql in insert_sql_list:
            print(sql)
            conn.execute(text(sql))
    print("Data written to MySQL table successfully!")


def test():
    data = {'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]}
    df = pd.DataFrame(data)
    table_name = 't2'
    # write(df, table_name)
    read()


if __name__ == "__main__":
    # Base = declarative_base()
    #
    # class User(Base):
    #     __tablename__ = 't2'
    #
    #     # 定义各字段
    #     id = Column(Integer, primary_key=True)
    #     name = Column(String)
    #     age = Column(Integer)
    #
    #     def __str__(self):
    #         return self.id
    init()
    test()
