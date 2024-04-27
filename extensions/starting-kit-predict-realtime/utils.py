import json
import os
import shutil
import hashlib
import pandas as pd
import numpy as np


def load_train_data(table_schema, paths):
    tables = {}
    for p in paths:
        for tn in table_schema["entity_names"]:
            cur_df = pd.read_parquet(os.path.join(p, tn))
            tables[tn] = pd.concat([tables[tn], cur_df]) if tn in tables else cur_df
    return tables


def copy_train_data(workspace_path, paths):
    for folder in paths:
        pathname = os.path.basename(folder)
        shutil.copytree(folder, os.path.join(workspace_path, pathname))


def load_table_schema(table_schema_path=None, workspace_path=None):
    if table_schema_path is None:
        table_schema_path = get_table_schema_in_workspace(workspace_path)
    with open(table_schema_path) as fp:
        table_schema = json.load(fp)
    return table_schema


def get_table_schema_in_workspace(workspace_path):
    return os.path.join(workspace_path, "table_schema.json")


def get_model_path_in_workspace(workspace_path):
    return os.path.join(workspace_path, "model.pickle")


def get_train_df_csv_in_workspace(workspace_path):
    return os.path.join(workspace_path, "train.csv")


def get_train_dir_in_workspace(workspace_path):
    workspace_train_path = os.path.join(workspace_path, "train")
    os.makedirs(workspace_train_path, exist_ok=True)
    return workspace_train_path


def get_task_type_in_workspace(workspace_path):
    return os.path.join(workspace_path, "task_type")


def get_create_table_sql_in_workspace(workspace_path):
    return os.path.join(workspace_path, "create_table_sql")


def get_window_sql_in_workspace(workspace_path):
    return os.path.join(workspace_path, "window_sql")


def md5_encode(item: str):
    md5_hash = int(
        hashlib.md5(item.encode()).hexdigest(), 16
    )  # 将十六进制的 MD5 哈希值转换为 int 类型
    md5_int64 = md5_hash & 0xFFFFFFFFFFFFFFFF
    return md5_int64


def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in df.columns:
        if df[col].dtype != object and df[col].dtype != "string":  # Exclude strings            
            # Print current column type
            # print("******************************")
            # print("Column: ",col)
            # print("dtype before: ",df[col].dtype)            
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            # print("min for this col: ",mn)
            # print("max for this col: ",mx)
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all():
                NAlist.append(col)
                df[col].fillna(mn - 1, inplace=True)

                # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True
                # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                        # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)

            # Print new column type
            # print("dtype after: ",df[col].dtype)
            # print("******************************")
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return df, NAlist
