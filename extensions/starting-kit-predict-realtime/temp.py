import pandas as pd
from sqlalchemy import create_engine, text


if __name__ == "__main__":
    data = {'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]}
    df = pd.DataFrame(data)
    engine = create_engine('mysql+pymysql://root:root@127.0.0.1:3307/db1', echo=True)
    # create table t2(id int, name string, age int);
    table_name = 't2'
    # with engine.connect() as conn:
    #     df.to_sql(table_name, conn.connection, if_exists='replace', index=False)
    df.to_sql(table_name, engine, if_exists='replace', index=False)

    # insert_sql_list = []
    # for index, row in df.iterrows():
    #     values_str = ', '.join([f"'{value}'" if isinstance(value, str) else str(value) for value in row])
    #     insert_sql = f"INSERT INTO {table_name} ({', '.join(df.columns)}) VALUES ({values_str});"
    #     insert_sql_list.append(insert_sql)
    # with engine.connect() as conn:
    #     for sql in insert_sql_list:
    #         print(sql)
    #         conn.execute(text(sql))
    print("Data written to MySQL table successfully!")
