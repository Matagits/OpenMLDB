import json
import logging
import os
import subprocess
import sys
import threading
import uuid

import pandas as pd
from flask import Flask, request

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s %(name)-12s %(levelname)-4s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__file__)

app = Flask(__name__)

table_schema = None
task_type = None
train_task_proc = {}
load_model_task_proc = {}
pred_task_proc = {}
thread_exception = {}
task_id = None
datapath_hist = []
data_info = None


@app.route("/automl/init", methods=["POST"])
def post_init_data():
    global table_schema, task_type
    req = request.get_json()
    table_schema = req["table_schema"]
    task_type = req["type"]
    return {}


@app.route("/automl/append_data", methods=["POST"])
def post_append_data():
    global datapath_hist, train_task_proc, task_id, table_schema, task_type
    req = request.get_json()

    # train_data
    if req["data"][0]["format"] == "parquet":
        datapath_hist.append(req["data"][0]["path"])
    else:
        return {
            "success": False,
            "message": f"Unsupport data format {req['data'][0]['format']}",
        }

    workspace_path = req["workspace_path"]
    fea_importance = req["feature_importance"]
    table_schema_path = "./table_schema.json"
    with open(table_schema_path, "w") as fp:
        json.dump(table_schema, fp)

    task_id = str(uuid.uuid4())
    train_task_proc[task_id] = subprocess.Popen(
        "python3.9 ./strategy2.py train %s %s %s %s %s %s"
        % (
            table_schema_path,
            task_type,
            workspace_path,
            fea_importance["feature_path"],
            fea_importance["feature_info_path"],
            " ".join([("'%s'" % p) for p in datapath_hist]),
        ),
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return {"success": True, "message": "", "data": {"handler": task_id}}


@app.route("/automl/get_status/<string:task_id>", methods=["GET"])
def get_status(task_id):
    global train_task_proc
    if task_id not in train_task_proc:
        return {
            "success": False,
            "message": "Handler %s not found" % task_id,
            "data": None,
        }
    proc = train_task_proc[task_id]
    retcode = proc.poll()
    if retcode is None:
        return {"success": True, "data": {"status": "Running"}}
    return {
        "success": True,
        "data": {"status": ("Success" if retcode == 0 else "Failed")},
    }


@app.route("/automl/load_model", methods=["POST"])
def load_model():
    def load_model_thread():
        global data_info, thread_exception
        from strategy2 import load_model_and_data

        try:
            data_info = load_model_and_data(workspace_path)
        except Exception as e:
            logger.error(f"Load Model Error, Exception {e}")
            thread_exception[task_id] = 1

    global task_id, workspace_path
    req = request.get_json()
    workspace_path = req["workspace_path"]
    task_id = str(uuid.uuid4())
    load_model_task_proc[task_id] = threading.Thread(target=load_model_thread)
    load_model_task_proc[task_id].start()
    return {"success": True, "message": "", "data": {"handler": task_id}}


@app.route("/automl/load_status/<string:task_id>", methods=["GET"])
def load_status(task_id):
    global load_model_task_proc
    if task_id not in load_model_task_proc:
        return {
            "success": False,
            "message": "Handler %s not found" % task_id,
            "data": None,
        }
    thread = load_model_task_proc[task_id]
    if thread.is_alive():
        return {"success": True, "data": {"status": "Running"}}
    elif task_id in thread_exception:
        return {"success": True, "data": {"status": "Failed"}}
    else:
        return {"success": True, "data": {"status": "Success"}}


@app.route("/automl/predict", methods=["POST"])
def predict_stage():
    from strategy2 import predict

    req = request.get_json()
    data = req["data"]

    pred_df = pd.DataFrame(data)
    table_schema = data_info["table_schema"]
    target_entity = table_schema["target_entity"]
    target_features = table_schema["entity_detail"][target_entity]["features"]
    target_feature_be_timestamp = [
        item["id"].split(".", 1)[1]
        for item in target_features
        if item["data_type"] == "Timestamp"
    ]
    for target_feature in target_feature_be_timestamp:
        pred_df[target_feature] = pred_df[target_feature].astype("datetime64[ns]")

    target_entity_index = table_schema["target_entity_index"]
    pred_result = predict(data_info, pred_df)
    result = []
    for _, item in pred_result.iterrows():
        result.append(
            {
                target_entity_index: item["reqId"],
                "value": float(item["score"]),
            }
        )
    # print(result)
    return {
        "success": True,
        "message": "",
        "data": result,
    }


if __name__ == "__main__":
    app.run("0.0.0.0", 80, threaded=True)
    # app.run("0.0.0.0", 29997, threaded=True)
    # app.run("0.0.0.0", 29996, threaded=True)
