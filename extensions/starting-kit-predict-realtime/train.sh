workspace_path=./work/workspace
feat_path=./work/feature.txt
feat_out_path=./work/feature_info.json
rm -rf ${workspace_path}
python strategy2.py train anta-sample/table_schema.json BinaryClassification ${workspace_path} ${feat_path} ${feat_out_path} anta-sample/0000
