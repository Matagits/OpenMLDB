import pandas as pd
import openmldb.dbapi


if __name__ == "__main__":
    db = openmldb.dbapi.connect(zk="0.0.0.0:2181", zkPath="/openmldb")
    cursor = db.cursor()
    cursor.execute("USE db1")
    table_name = "test"
    partition_by_col = "eventTime_weekday"
    order_by_col = "clickToday"
    cols = ["showCount1d", "duration"]
    agg_cols = ", ".join(map(lambda col_name: f"max({col_name}) OVER w AS {col_name}_max", cols))
    sql = f"SELECT {agg_cols} FROM {table_name} " \
          f"WINDOW w AS (PARTITION BY {partition_by_col} ORDER BY {order_by_col} " \
          f"ROWS BETWEEN 50 PRECEDING AND CURRENT ROW);"
    print("sql: " + sql)
    result = cursor.execute(sql)
    all1 = result.fetchall()
    df = pd.DataFrame(all1, columns=cols)
    print(df.head())
    print("Data written to MySQL table successfully!")
