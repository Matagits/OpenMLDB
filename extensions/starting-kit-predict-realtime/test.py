import time

import openmldb.dbapi
from logger import logger


def test():
    global db, cursor
    retry = 5
    while retry > 0:
        try:
            db = openmldb.dbapi.connect(zk="0.0.0.0:2181", zkPath="/openmldb")
            break
        except Exception as e:
            logger.warn(e)
            logger.warn(retry)
            time.sleep(2)
            retry -= 1
    cursor = db.cursor()
    try:
        cursor.execute("CREATE DATABASE db1")
    except openmldb.dbapi.dbapi.DatabaseError as e:
        logger.warn(e)
    cursor.execute("USE db1")
    cursor.execute("set @@execute_mode='offline';")


if __name__ == "__main__":
    test()
