import glob
import os
import time

import openmldb.dbapi
import logging
import pandas as pd
from numpy import dtype

from autofe import OpenMLDBSQLGenerator, AutoXTrain
from utils import get_create_table_sql_in_workspace, get_index_sql_in_workspace, get_window_sql_in_workspace, \
    get_top_features_in_workspace, get_train_df_csv_in_workspace
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s %(name)-12s %(levelname)-4s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__file__)


class OpenMLDBHelper:
    def __init__(self, workspace):
        logger.info(f"OpenMLDBHelper init")
        self.db_name = "db1"
        self.table_name = "automl"
        self.workspace_path = workspace
        self.online_mode = False
        retry = 60
        while retry > 0:
            try:
                self.db = openmldb.dbapi.connect(zk="0.0.0.0:2181", zkPath="/openmldb")
                break
            except Exception as e:
                logger.warning(e)
                logger.info(f"left retry: {retry}")
                time.sleep(5)
                retry -= 1
        self.cursor = self.db.cursor()
        try:
            self.cursor.execute(f"CREATE DATABASE {self.db_name}")
        except openmldb.dbapi.dbapi.DatabaseError as e:
            logger.warning(e)
        self.cursor.execute(f"USE {self.db_name}")
        self.cursor.execute(f"set @@spark_config='spark.driver.memory=4g';")
        logger.info(f"Finished init")

    def _try_drop_table(self):
        try:
            self.cursor.execute(f"DROP TABLE {self.table_name}")
        except openmldb.dbapi.dbapi.DatabaseError as e:
            logger.warning(e)

    def _load_data_infile(self, table: str, file_path: str, mode: str = 'overwrite', block: bool = True):
        file_path = f"file://{os.path.abspath(file_path)}"
        deep_copy = "true" if self.online_mode else "false"
        if self.online_mode:
            mode = "append"
        self.cursor.execute(
            f"LOAD DATA INFILE '{file_path}' INTO TABLE {table} options(format='csv', mode='{mode}', deep_copy={deep_copy});")
        job_id = None
        for job_info in self.cursor.fetchall():
            job_id = job_info[0]
        if block:
            job_state = self._get_job_state(job_id)
            while job_state.lower() not in ['failed', 'finished', 'killed']:
                logger.info(f'load data infile job {job_id} state: {job_state}')
                time.sleep(5)
                job_state = self._get_job_state(job_id)
            logger.info(f'load data infile job {job_id} state: {job_state}')
        return job_id

    def _get_job_state(self, job_id: str) -> str:
        self.cursor.execute(f"show job {job_id}")
        for job_info in self.cursor.fetchall():
            return job_info[2]

    def append_window_union_features(self, df, label, number_cols, id_col, partition_by_col, sort_by_col):
        # print(df.head())
        # print(df[[id_col, partition_by_col, sort_by_col]].head())

        logger.info("Start to write df to openmldb")
        self.write_df_to_openmldb(df, self.table_name)
        logger.info("Finished to write df to openmldb")
        return self.append_features(df, label, self.table_name, number_cols, id_col, partition_by_col, sort_by_col)

    def write_df_to_openmldb(self, df_origin, table):
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

        self._create_table(table, col_infos)
        self._write_to_db(df, table)

    def _create_table(self, table, col_infos):
        pass

    def _write_to_db(self, df, table):
        pass

    def append_features(self, df, label, table, cols, id_col, partition_by_col, order_by_col):
        return None


class TrainHelper(OpenMLDBHelper):
    def __init__(self, workspace):
        super().__init__(workspace)
        self.online_mode = False
        self.cursor.execute(f"set @@execute_mode='offline';")
        logger.info(f"Train finished init")

    def _create_table(self, table, col_infos):
        self._try_drop_table()

        sql = f"CREATE TABLE {table} ({', '.join(col_infos)})"
        self.cursor.execute(sql)
        create_table_sql_path = get_create_table_sql_in_workspace(self.workspace_path)
        with open(create_table_sql_path, "w") as fp:
            fp.write(sql)
        logger.info(f"Write CreateTableSql to {create_table_sql_path}")

    def _write_to_db(self, df, table):
        start_time = time.time()
        train_df_csv_path = get_train_df_csv_in_workspace(self.workspace_path)
        df.to_csv(train_df_csv_path, index=False, header=True, encoding="utf-8")
        self._load_data_infile(table, train_df_csv_path)
        end_time = time.time()

        logger.info("write_to_db cost time: " + str(end_time - start_time))
        logger.info("Data written to openMLDB successfully!")

    def append_features(self, df, label, table, cols, id_col, partition_by_col, order_by_col):
        agg_cols = []
        logger.info("autofe")
        conf = {
            'tables': [
                {
                    'table': table
                }
            ],
            'main_table': table,
            'windows': [
                {
                    'name': 'w1',
                    'partition_by': partition_by_col,
                    'order_by': order_by_col,
                    'window_type': 'rows_range',
                    'start': '1d PRECEDING',
                    'end': 'CURRENT ROW'
                },
                {
                    'name': 'w2',
                    'partition_by': partition_by_col,
                    'order_by': order_by_col,
                    'window_type': 'rows_range',
                    'start': '5d PRECEDING',
                    'end': 'CURRENT ROW'
                }
            ]
        }
        offline_feature_path = '/tmp/automl_offline_feature'

        sql_generator = OpenMLDBSQLGenerator(conf, df)
        sql, feature_path = sql_generator.time_series_feature_sql()
        logger.error(f"time_series_feature_sql: {sql}")

        top_features_with_id = agg_cols.copy()
        top_features_with_id.insert(0, id_col)
        df_append_features = self._get_append_features_df(sql, feature_path)
        df_append_features_ordered = pd.merge(df[id_col], df_append_features, on=id_col, how="left")

        label.index = df_append_features_ordered.index
        label_col_name = 'automl_label'
        df_append_features_ordered[label_col_name] = label

        train_set, test_set_with_y = train_test_split(df_append_features_ordered, train_size=0.8)
        test_set = test_set_with_y.drop(columns=[label_col_name])

        # save for backup
        train_name = 'train.parquet'
        test_name = 'test.parquet'
        train_set.to_parquet(offline_feature_path + '/' + train_name, index=False)
        test_set.to_parquet(offline_feature_path + '/' + test_name, index=False)

        id_list = [id_col]
        topk = 30
        logger.error(f'get top {topk} features')
        topk_features = AutoXTrain(debug=False).get_top_features(
            train_set, test_set, id_list, label_col_name, offline_feature_path, topk)
        logger.error(f'top {len(topk_features)} feas: {topk_features}')

        # decode feature to final sql
        final_sql = sql_generator.decode_time_series_feature_sql_column(
            id_col, topk_features)
        logger.error(f'final sql: {final_sql}')

        top_features_with_id = topk_features.copy()
        top_features_with_id.insert(0, id_col)
        df_final = pd.merge(df, df_append_features_ordered[top_features_with_id], on=id_col, how="left")
        agg_cols.extend(topk_features)
        logger.error(f'agg_cols: {agg_cols}')

        top_features_path = get_top_features_in_workspace(self.workspace_path)
        with open(top_features_path, "w") as fp:
            fp.write(str(topk_features))

        window_sql_path = get_window_sql_in_workspace(self.workspace_path)
        with open(window_sql_path, "w") as fp:
            fp.write(final_sql)

        index_sql_path = get_index_sql_in_workspace(self.workspace_path)
        with open(index_sql_path, "w") as fp:
            fp.write(f"index(key={partition_by_col}, ttl=50, ttl_type=latest, ts=`{order_by_col}`)")

        df_final[topk_features] = df_final[topk_features].fillna(0)
        return df_final, topk_features

    def _get_append_features_df(self, sql, feature_path):
        self._export_new_feature_outfile(sql)

        def remove_prefix(text, prefix): return text[len(
            prefix):] if text.startswith(prefix) else text

        feature_path = remove_prefix(feature_path, 'file://')

        logging.info(f'load {feature_path}')
        df = pd.concat(map(pd.read_parquet, glob.glob(
            os.path.join('', feature_path + '/*.parquet'))))
        return df

    def _export_new_feature_outfile(self, sql: str, block: bool = True):
        self.cursor.execute(sql)
        job_id = None
        for job_info in self.cursor.fetchall():
            job_id = job_info[0]
        if block:
            job_state = self._get_job_state(job_id)
            while job_state.lower() not in ['failed', 'finished', 'killed']:
                logger.info(f'export new feature outfile job {job_id} state: {job_state}')
                time.sleep(5)
                job_state = self._get_job_state(job_id)
            logger.info(f'export new feature outfile job {job_id} state: {job_state}')
        return job_id


class PredictHelper(OpenMLDBHelper):
    def __init__(self, workspace):
        super().__init__(workspace)
        self.online_mode = True
        self.cursor.execute(f"set @@execute_mode='online';")
        try:
            self._try_drop_table()

            create_table_sql_path = get_create_table_sql_in_workspace(self.workspace_path)
            logger.info(f"Load CreateTableSql From {create_table_sql_path}")
            with open(create_table_sql_path, "r") as fp:
                create_table_sql = fp.read()
            logger.info(f"CreateTableSql: {create_table_sql}")

            index_sql_path = get_index_sql_in_workspace(self.workspace_path)
            logger.info(f"Load IndexSql From {index_sql_path}")
            with open(index_sql_path, "r") as fp:
                index_sql = fp.read()
            create_table_sql = f"{create_table_sql[:-1]}, {index_sql})"
            logger.info(f"new create: {create_table_sql}")

            try:
                self.cursor.execute(create_table_sql)
            except openmldb.dbapi.dbapi.DatabaseError as e:
                logger.warning(e)

            train_df_csv_path = get_train_df_csv_in_workspace(self.workspace_path)
            self._load_data_infile(self.table_name, train_df_csv_path)
            logger.info("Finished loading train data")

            window_sql_path = get_window_sql_in_workspace(self.workspace_path)
            logger.info(f"Load WindowSql From {window_sql_path}")
            with open(window_sql_path, "r") as fp:
                self.window_sql = fp.read()
            logger.info(f"WindowSql: {self.window_sql}")
        except Exception as ex:
            logger.warning(ex)
        logger.info(f"Predict finished init")

    def _write_to_db(self, df, table):
        start_time = time.time()

        cols = ', '.join(df.columns)
        # df.apply(lambda x: self._insert(x, table, cols), axis=1)

        # logger.error(f"write df length: {len(df)}")
        temp = df.apply(lambda x: str(tuple(x)), axis=1)
        temp_list = temp.values.tolist()
        # logger.error(f"value length: {len(temp_list)}")
        self.cursor.execute(f"INSERT INTO {table} ({cols}) VALUES {', '.join(temp_list)};")
        # self.cursor.execute(f"INSERT INTO {table} ({cols}) VALUES {', '.join(temp_list[:50])};")
        # self.cursor.execute(f"INSERT INTO {table} ({cols}) VALUES {', '.join(temp_list[50:])};")
        # logger.error(f"result: {result}")
        # logger.error(f"result fetchall: {result.fetchall()}")

        end_time = time.time()

        # df.sort_values(by=["eventTime", 'reqId'], ascending=[True, True], inplace=True)
        # temp1 = df.apply(lambda x: str(tuple(x[['reqId', 'eventTime']])), axis=1)
        # df_str = '\n'.join(temp1.values.tolist())
        # logger.error(f"df: \n{df_str}")

        logger.info("write_to_db cost time: " + str(end_time - start_time))
        logger.info("Data written to openMLDB successfully!")

    def _insert(self, row, table, cols):
        values_str = ', '.join([f"'{value}'" if isinstance(value, str) else str(value) for value in row])
        insert_sql = f"INSERT INTO {table} ({cols}) VALUES ({values_str});"
        self.cursor.execute(insert_sql)

    def append_features(self, df, label, table, cols, id_col, partition_by_col, order_by_col):
        start_time = time.time()
        top_features_path = get_top_features_in_workspace(self.workspace_path)
        with open(top_features_path, "r") as fp:
            top_features = eval(fp.read())
        df[top_features] = df.apply(self._request_append_features, axis=1)
        end_time = time.time()
        logger.info("get window union features cost time: " + str(end_time - start_time))
        # logger.error(f"df: \n{df.head()}")

        df[top_features] = df[top_features].fillna(0)
        # logger.error(f"fill na df: \n{df.head()}")
        return df, top_features

    def _request_append_features(self, row):
        value_tuple = tuple(row)
        result = self.cursor.execute(f"{self.window_sql} CONFIG (execute_mode = 'request', values = {value_tuple})")
        res_tuple = result.fetchone()
        return pd.Series(res_tuple[1:])


default_workspace = ""
train_mode = True
helper = None


def init(workspace, is_train_mode):
    global default_workspace, train_mode, helper
    default_workspace = workspace
    train_mode = is_train_mode
    if train_mode:
        helper = TrainHelper(default_workspace)
    else:
        helper = PredictHelper(default_workspace)


def get_helper():
    return helper
