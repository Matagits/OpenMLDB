workspace_path=./work/workspace
feat_path=./work/feature.txt
feat_out_path=./work/feature_info.json
data_path=/tmp/ybw/automl/judge_flow/data/at-20240325/0000
data_schema_path=/tmp/ybw/automl/judge_flow/data/at-20240325/table_schema.json
rm -rf ${workspace_path}
python strategy2.py train ${data_schema_path} BinaryClassification ${workspace_path} ${feat_path} ${feat_out_path} ${data_path}