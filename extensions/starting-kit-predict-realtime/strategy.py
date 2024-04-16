import json
import os
import pickle
import sys

import pandas as pd

from logger import logger
from utils import (
    md5_encode,
    copy_train_data,
    get_model_path_in_workspace,
    get_table_schema_in_workspace,
    get_task_type_in_workspace,
    get_train_dir_in_workspace,
    load_table_schema,
    load_train_data,
)


"""
训练

输入：
table_schema: 表结构，包含每张表的表名、字段名，字段类型及表关系等
typ: 任务类型, BinaryClassification/Regression
tables: 数据dict, 格式：{'表名': pandas.DataFrame}

返回：
所有预测阶段需要的信息，例如模型、词表、超参等等。注意训练和预测只能通过本返回信息进行数据传递
"""


# TODO
def train(table_schema, typ, tables):
    # join label
    insid_col = table_schema["target_entity_index"]
    label_col = table_schema["target_label"].split(".", 1)[1]
    Y = (
        tables[table_schema["target_entity"]]
        .merge(tables["action"], on=insid_col, how="left")[label_col]
        .fillna(0.0)
        .astype(float)
    )
    # train
    if typ == "BinaryClassification":
        from pipeline import FeatureEngineerInitTransformer
        from pipeline import FeatureInfoSave
        from pipeline import FeatureEngineerTransformer

        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        # 全部的 step
        full_pipeline = Pipeline(steps=[
            ('feature_engineer_first', FeatureEngineerInitTransformer()),
            ('feature_info_save', FeatureInfoSave()),
            ('feature_engineer_second', FeatureEngineerTransformer()),
            ('classifier', LogisticRegression())
        ])
        # 训练除了 classifier 之外的 Pipeline 部分
        preprocessor_pipeline = Pipeline(full_pipeline.steps[:-1])
        X_transformed = preprocessor_pipeline.fit_transform((table_schema, tables))
        # 对 classifier 进行训练
        classifier:LogisticRegression = full_pipeline.named_steps['classifier']
        classifier.fit(X_transformed, Y)
        # 基于训练集的预测结果，用于后续特征重要性处理
        train_pred_reqId = tables[table_schema["target_entity"]]["reqId"]
        train_pred_label = classifier.predict_proba(X_transformed)[:, 1]
        train_pred_mapping = {
            str(md5_encode(reqId)): label
            for reqId, label in zip(train_pred_reqId, train_pred_label)
        }
        feature_path = globals().get('feature_path', None)
        if feature_path is not None:
            with open(feature_path, "w") as fp:
                with open(feature_path+"_temp", "r") as temp_fp:
                    for line in temp_fp:
                        reqId = line.split("|", 1)[0]
                        label = train_pred_mapping.get(reqId)
                        fp.write(f"{label} {line}")
    elif typ == "Regression":
        raise Exception("unsupport")
    else:
        raise Exception("Unknown task type")
    # 返回所有predict阶段需要的数据，（注意：无法使用全局变量进行数据传递）
    return {"model": full_pipeline}


"""
加载模型与数据

输入：
workspace_path：模型、数据的加载根文件夹
默认的 table-schema、训练数据、模型文件、task 类型的保存位置，参考 utils.py 中 get_xxx_in_workspace

返回：
所有预测阶段需要的信息，例如模型、词表、超参等等。
这部分数据会直接作为 predict 接口中 data_info 参数传入
"""


# TODO
def load_model_and_data(workspace_path):
    table_schema_path = get_table_schema_in_workspace(workspace_path)
    logger.info(f"Load Table Schema From {table_schema_path}")
    with open(table_schema_path, "r") as fp:
        table_schema = json.load(fp)

    task_type_path = get_task_type_in_workspace(workspace_path)
    logger.info(f"Load TaskType From {task_type_path}")
    with open(task_type_path, "r") as fp:
        task_type = fp.read()

    model_path = get_model_path_in_workspace(workspace_path)
    logger.info(f"Load Model From {model_path}")
    with open(model_path, "rb") as fp:
        model = pickle.load(fp)

    train_dir = get_train_dir_in_workspace(workspace_path)
    train_paths = [
        os.path.join(train_dir, dirname) for dirname in os.listdir(train_dir)
    ]
    hist_tables = load_train_data(table_schema, train_paths)
    # 其他需要对数据的预先处理

    return {
        "table_schema": table_schema,  # 需要包含
        "task_type": task_type,  # 其他可能需要的数据
        "model": model,
        "tables": hist_tables,
    }


"""
预测

输入：
data_info: load_model_and_data 函数返回的内容
pred_df: 待评估的数据，pandas.DataFrame格式

输出：
预测结果DataFrame，须包含如下字段：
    reqId(String)，待评估数据的id，可从待评估数据中获取
    score(double)，预测打分结果
"""


# TODO
def predict(data_info, pred_df):
    # 解析 data_info
    table_schema = data_info["table_schema"]
    task_type = data_info["task_type"]
    model = data_info["model"]
    tables = data_info["tables"]
    tables[table_schema["target_entity"]] = pred_df

    insid_col = table_schema["target_entity_index"]
    result = pd.DataFrame()
    result[insid_col] = pred_df[insid_col]
    if task_type == "BinaryClassification":
        prob = model.predict_proba((table_schema, tables))
        result["score"] = prob[:,1]
    elif task_type == "Regression":
        raise Exception("unsupport")
    else:
        raise Exception("Unknown task type")
    return result


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        logger.info("Start training ...")
        table_schema_path = sys.argv[2]
        task_type = sys.argv[3]
        workspace_path = sys.argv[4]
        feature_path = sys.argv[5]
        feature_info_path = sys.argv[6]
        train_paths = sys.argv[7:]

        os.makedirs(workspace_path, exist_ok=True)

        logger.info(f"Loading table schema from {table_schema_path}")
        table_schema = load_table_schema(table_schema_path)

        # save table_schema to workspace
        table_schema_path = get_table_schema_in_workspace(workspace_path)
        with open(table_schema_path, "w") as fp:
            json.dump(table_schema, fp)

        # save task_type to workspace
        task_type_path = get_task_type_in_workspace(workspace_path)
        with open(task_type_path, "w") as fp:
            fp.write(task_type)

        # save train data in workspace
        train_dir = get_train_dir_in_workspace(workspace_path)
        logger.info(f"Copying Data from {train_paths} to {train_dir}")
        copy_train_data(train_dir, train_paths)

        # load train data
        logger.info(f"Loading tables from {train_paths}")
        tables = load_train_data(table_schema, train_paths)

        os.environ['feature_path'] = feature_path
        os.environ['feature_info_path'] = feature_info_path

        # train model
        logger.info("Training")
        train_result = train(table_schema, task_type, tables)
        model = train_result["model"]

        # save model
        model_path = get_model_path_in_workspace(workspace_path)
        logger.info(f"Saving model to {model_path}")
        with open(model_path, "wb") as fp:
            pickle.dump(model, fp)
        logger.info("Train finished")
    elif cmd == "load_and_predict":  # 该部分仅仅用于本地调试
        logger.info("Start Load and Predict ...")
        workspace_path = sys.argv[2]
        eval_path = sys.argv[3]

        # load model and data info
        data_info = load_model_and_data(workspace_path)

        # load predict data offline
        logger.info(f"Loading predict data from {eval_path}")
        logger.info(f"Loading predict data from {eval_path}")
        pred_df = pd.read_parquet(eval_path)

        # predict data
        result = predict(data_info, pred_df)

        # save result to local
        logger.info(f"Saving predict result to pred.csv")
        result.to_csv("pred.csv", header=True, index=False)
        logger.info("Predict finished")
        logger.info(f"Saving predict result to pred.csv")
        result.to_csv("pred.csv", header=True, index=False)
        logger.info("Predict finished")
    else:
        raise Exception("Unkonwn cmd")
