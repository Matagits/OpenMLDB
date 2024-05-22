workspace_path=./work/workspace
feat_path=./work/feature.txt
feat_out_path=./work/feature_info.json
data_path=/root/test/data/anta-sample/0000
data_schema_path=/root/test/data/anta-sample/table_schema.json
rm -rf ${workspace_path}
python3 strategy2.py train ${data_schema_path} BinaryClassification ${workspace_path} ${feat_path} ${feat_out_path} ${data_path}