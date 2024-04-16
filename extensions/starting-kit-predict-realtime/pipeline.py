import json
import os
from utils import md5_encode
from typing import Dict, List, Tuple
import pandas as pd
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class DictOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self
    """
    从上面展开为下面的形式
                    feature1                  feature2
    0  {'key1': 1, 'key2': 2}              {'keyA': 10}
    1  {'key2': 2, 'key3': 3}  {'keyA': 20, 'keyB': 30}
    2  {'key1': 1, 'key3': 3}              {'keyB': 20}
    
    feature1_key2  feature1_key3  feature1_key1  feature2_keyB  feature2_keyA
    0              2              0              0              0             10
    1              2              3              0             30             20
    2              0              3              1             20              0

    在 predict 的时候, 如果遇到之前没出现过的 key, 那么会自动忽略。
    """
    def fit(self, X, y=None):
        self.keys_ = {}
        for col in X.columns:
            if isinstance(X[col].iloc[0], dict):
                self.keys_[col] = list(set(key for item in X[col].dropna() for key in item))
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col, keys in self.keys_.items():
            col_expanded = {}
            for key in keys:
                new_col_name = f"{col}_{key}"
                col_expanded[new_col_name] = [(d[key] if isinstance(d, dict) and key in d else 0) for d in X[col]]
            X_transformed = X_transformed.join(pd.DataFrame(col_expanded))
            X_transformed = X_transformed.drop(col, axis=1)
        return X_transformed
    
    def get_feature_names_out(self, columns: List[str]):
        return [
            f"{col}_{key}"
            for col in columns
            for key in self.keys_[col]
        ]
    


"""
特征初步处理，将所有的初始特征转换为三类特征，为了后续平台侧进行特征重要性评估：
- number 类型：例如年龄、商品价格。
- category 类型：例如省份、国家。
- 多值类型：例如词袋模型，文本转换为多值特征。
    title1: 火车车厢车厢 -> {火车:1, 车厢:2}
    title2: 汽车车厢 -> {汽车:1, 车厢:1}

transform:
    返回特征初步处理的结果 + feature_info 信息
"""

class FeatureEngineerInitTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.category_columns = None

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> Tuple[pd.DataFrame, Dict]:
        """
        Args:
            X: [table_schema: Dict, tables: List[pd.DataFrame]]
        return:
            [df:pd.DataFrame, feat_info: Dict]
            df: 基于所有的 tables 生成的初步特征结果
            feat_info: 每一个特征列的信息
        """
        table_schema, tables = X
        df = pd.DataFrame()
        feature_info = {}
        target_entity_table = tables[table_schema["target_entity"]]
        for c in table_schema["entity_detail"][table_schema["target_entity"]]["features"]:
            col_name = c["id"].split(".", 1)[1]
            if c["skip"]:
                continue
            if col_name == "reqId": # 保留 reqId，在 FeatureInfoSave 需要进行数据存储
                df[col_name] = target_entity_table[col_name]
            elif c["feature_type"] == "Int":
                df[col_name] = target_entity_table[col_name].fillna(0).infer_objects(copy=False).astype(int)
                feature_info[col_name] = {
                    "feature_description": f"raw feature of {col_name}",
                    "type": "Number",
                }
            elif c["feature_type"] == "Double":
                df[col_name] = target_entity_table[col_name].fillna(0.0).astype(float)
                feature_info[col_name] = {
                    "feature_description": f"raw feature of {col_name}",
                    "type": "Number",
                }
            elif c["feature_type"] == "Timestamp":
                col_values = target_entity_table[col_name].astype('datetime64[ns]')
                col_values = col_values.fillna(pd.Timestamp('2021-01-03 07:25:00'))
                df[col_name + "_month"] = col_values.dt.month
                df[col_name + "_weekday"] = col_values.dt.weekday
                df[col_name + "_month"] = df[col_name + "_month"].apply(str).astype("category")
                df[col_name + "_weekday"] = df[col_name + "_weekday"].apply(str).astype("category")
                feature_info[col_name + "_weekday"] = {
                    "feature_description": f"Generate from {col_name}, get weekday of this col.",
                    "type": "Category",
                }
                feature_info[col_name + "_month"] = {
                    "feature_description": f"Generate from {col_name}, get month of this col.",
                    "type": "Category",
                }
            # 调试的时候训练太慢了，关了
            elif c["data_type"] == "ArrayString(,)":
                col_values = target_entity_table[col_name].fillna("")
                df[col_name + "_multi_value"] = col_values.apply(lambda x: Counter(x.split(",")))
                feature_info[col_name + "_multi_value"] = {
                    "feature_description": f"Generate from {col_name}, multi-value feature, using Counter(x.split(',')).",
                    "type": "Multi-Value",
                }
            elif c["feature_type"] == "String":
                df[col_name] = target_entity_table[col_name].fillna("").astype("category")
                feature_info[col_name] = {
                    "feature_description": f"raw feature of {col_name}",
                    "type": "Category",
                }
        # print(df.head())
        return df, feature_info


"""
生成特征数据，用于之后平台侧进行特征重要性计算、给客户展示特征等。
fit 部分，存储详细的 feature-info, 同时存储训练时的各个特征值
transform 阶段，对 dataframe 数据进行过滤，只留取 feature_info 中配置的列
"""

class FeatureInfoSave(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.feat_col_name_list = []
    
    def fit(self, X, y=None):
        table, feat_info = X
        
        # 获取确认保留的数据列
        for feat_col_name, feat_col_info in feat_info.items():
            if feat_col_info["type"] == "Number":
                if pd.api.types.is_numeric_dtype(table[feat_col_name]):
                    self.feat_col_name_list.append(feat_col_name)
            elif feat_col_info["type"] == "Category":
                self.feat_col_name_list.append(feat_col_name)
            else:
                self.feat_col_name_list.append(feat_col_name)

        # 从 global 中获取 feature_info 的保存路径
        feature_path = os.environ.get('feature_path', None)
        feature_info_path = os.environ.get('feature_info_path', None)
        if feature_info_path is None or feature_path is None:
            return self

        # 将 feature info 保存下来
        feature_full_info = {str(idx): {"col_name": col, **feat_info[col]}
                             for idx, col in enumerate(self.feat_col_name_list)}
        with open(feature_info_path, "w") as fp:
            json.dump(feature_full_info, fp, indent=2, ensure_ascii=False)

        # 存储训练时的各个特征值
        feat_col_to_idx_mapping = {col: idx for idx, col in enumerate(self.feat_col_name_list)}
        with open(feature_path+"_temp", "w") as fp:
            for _, row_value in table.iterrows():
                feat_item = f"{md5_encode(row_value['reqId'])}|"
                for col_name, col_value in row_value.items():
                    if col_name in ["label", "reqId"]:
                        continue
                    feat_idx = feat_col_to_idx_mapping.get(col_name)
                    if feat_info[col_name]["type"] == "Number":
                        feat_item += f" {feat_idx}:1:{col_value}"
                    elif feat_info[col_name]["type"] == "Category":
                        feat_item += f" {feat_idx}:{md5_encode(str(col_value))}"
                    else:
                        for a, b in col_value.items():
                            feat_item += f" {feat_idx}:{md5_encode(str(a))}:{b}"
                fp.write(feat_item + "\n")
        return self

    def transform(self, X) -> Tuple[pd.DataFrame, Dict]:
        table, feat_info = X
        return table[self.feat_col_name_list], feat_info

"""
特征进一步处理
- 数值类型，保持不变
- category 类型转换为 one-hot 形式
- 若某一列均为列表形式，那么则视为多值类型，根据 key 进行展开，参考 DictOneHotEncoder。
"""

class FeatureEngineerTransformer:
    def __init__(self):
        self.category_columns = None
        self.category_encoder = OneHotEncoder(sparse_output = False, handle_unknown='ignore')
        self.multi_category_columns = None
        self.multi_value_encoder = DictOneHotEncoder()

    def fit(self, X: Tuple[pd.DataFrame, Dict], y=None):
        table, feat_info = X
        # category 类型的列处理
        self.category_columns = [
            col_name for col_name, col_info in feat_info.items() if col_info["type"] == "Category"]
        self.category_encoder.fit(table[self.category_columns])
        # print("category_columns", self.category_columns)

        # multi-value 类型的列处理
        self.multi_category_columns = [
            col_name for col_name, col_info in feat_info.items() if col_info["type"] == "Multi-Value"]
        self.multi_value_encoder.fit(table[self.multi_category_columns])
        return self

    def transform(self, X: Tuple[pd.DataFrame, Dict]) -> pd.DataFrame:
        table, feat_info = X
        # category 类型的列处理
        table_encoded = pd.DataFrame(
            self.category_encoder.transform(table[self.category_columns]),
            columns=self.category_encoder.get_feature_names_out(self.category_columns),
            index = table.index
        )
        # multi-value 类型的列处理
        table_multi_value = pd.DataFrame(
            self.multi_value_encoder.transform(table[self.multi_category_columns]),
            columns=self.multi_value_encoder.get_feature_names_out(self.multi_category_columns),
            index = table.index
        )
        table.drop(columns=self.multi_category_columns + self.category_columns, inplace=True)
        table = pd.concat([table, table_encoded, table_multi_value], axis=1)
        return table
