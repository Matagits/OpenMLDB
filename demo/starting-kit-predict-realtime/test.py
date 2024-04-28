import openmldb.dbapi


def test():
    db = openmldb.dbapi.connect(zk="0.0.0.0:2181", zkPath="/openmldb")
    cursor = db.cursor()
    cursor.execute("USE db1")
    cursor.execute(f"set @@execute_mode='online';")
    result = cursor.callproc("d2", (1, 1, 'a'))
    out_schema = result.get_resultset_schema()
    print(out_schema)
    all_result = result.fetchone()
    print(type(all_result))
    print("all_result: " + str(all_result))


if __name__ == "__main__":
    test()
