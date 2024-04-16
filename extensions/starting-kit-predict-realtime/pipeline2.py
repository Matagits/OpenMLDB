import json
import os
import time

import numpy as np
from utils import md5_encode, reduce_mem_usage
from typing import Dict, List, Tuple
import pandas as pd
from collections import Counter, defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2


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
        col_expanded = np.empty((0, X.shape[0]))
        for col, keys in self.keys_.items():
            for key in keys:
                col_expanded = np.append(col_expanded, 
                                         [np.array([(d[key] if isinstance(d, dict) and key in d else 0) for d in X[col]])],
                                         axis=0)
        return col_expanded.T
    
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

    def transform(self, X) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Args:
            X: [table_schema: Dict, tables: List[pd.DataFrame], label: Optional[pd.DataFrame]]
        return:
            [df:pd.DataFrame, label:pd.DataFrame, feat_info: Dict]
            df: 基于所有的 tables 生成的初步特征结果
            label: reqId 对应的 label
            feat_info: 每一个特征列的信息
        """
        table_schema, tables, label = X
        df = pd.DataFrame()
        feature_info = {}
        target_entity_table = tables[table_schema["target_entity"]]
        # print(target_entity_table.info())
        for c in table_schema["entity_detail"][table_schema["target_entity"]]["features"]:
            col_name = c["id"].split(".", 1)[1]
            if c["skip"]:
                continue
            if col_name == "reqId": # 保留 reqId，在 FeatureInfoSave 需要进行数据存储
                df[col_name] = target_entity_table[col_name].astype(str)
            elif c["feature_type"] == "Int" or c["feature_type"] == "BigInt":
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
                df[col_name + "_month"] = df[col_name + "_month"].apply(str)
                df[col_name + "_weekday"] = df[col_name + "_weekday"].apply(str)
                feature_info[col_name + "_weekday"] = {
                    "feature_description": f"Generate from {col_name}, get weekday of this col.",
                    "type": "Category",
                }
                feature_info[col_name + "_month"] = {
                    "feature_description": f"Generate from {col_name}, get month of this col.",
                    "type": "Category",
                }
            # 调试的时候训练太慢了，关了
            # elif c["data_type"] == "ArrayString(,)":
            #     col_values = target_entity_table[col_name].fillna("").astype(str)
            #     df[col_name + "_multi_value"] = col_values.apply(lambda x: Counter(x.split(",")))
            #     feature_info[col_name + "_multi_value"] = {
            #         "feature_description": f"Generate from {col_name}, multi-value feature, using Counter(x.split(',')).",
            #         "type": "Multi-Value",
            #     }
            # 用 one-hot 的话，内存会爆掉，后续有时间试一下 lgbm
            # elif c["feature_type"] == "String":
            #     # 去掉外键
            #     skip = False
            #     for relation in table_schema["relations"]:
            #         if col_name in relation["from_entity_keys"]:
            #             skip = True
            #             break
            #     if skip: continue
            #     df[col_name] = target_entity_table[col_name].fillna("")
            #     feature_info[col_name] = {
            #         "feature_description": f"raw feature of {col_name}",
            #         "type": "Category",
            #     }
        # print(df.head())
        return df, label, feature_info


"""
特征进一步处理
- 数值类型，保持不变
- category 类型转换为 one-hot 形式
- 若某一列均为列表形式，那么则视为多值类型，根据 key 进行展开，参考 DictOneHotEncoder。
"""

class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.category_columns = None
        self.category_encoder = OneHotEncoder(sparse_output = False, handle_unknown='ignore')
        self.multi_value_columns = None
        self.multi_value_encoder = DictOneHotEncoder()

    def fit(self, X: Tuple[pd.DataFrame, pd.DataFrame, Dict], y=None):
        table, _, feat_info = X
        # category 类型的列处理
        self.category_columns = [
            col_name for col_name, col_info in feat_info.items() if col_info["type"] == "Category"]
        self.category_encoder.fit(table[self.category_columns])
        # print("category_columns", self.category_columns)

        # multi-value 类型的列处理
        self.multi_value_columns = [
            col_name for col_name, col_info in feat_info.items() if col_info["type"] == "Multi-Value"]
        self.multi_value_encoder.fit(table[self.multi_value_columns])
        print("fit done")
        return self

    def transform(self, X: Tuple[pd.DataFrame, pd.DataFrame, Dict]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        table, label, feat_info = X
        # category 类型的列处理
        table_encoded = pd.DataFrame(
            self.category_encoder.transform(table[self.category_columns]),
            columns=self.category_encoder.get_feature_names_out(self.category_columns),
            index = table.index,
            dtype=np.int8
        )
        # 记录下原始 category 列名 => 新的一批 category 列名
        for idx, category_column in enumerate(self.category_columns):
            feat_info[category_column]["colname_transfer"] = [
                f"{category_column}_{cat}" for cat in self.category_encoder.categories_[idx]]

        # multi-value 类型的列处理
        table_multi_value = pd.DataFrame(
            self.multi_value_encoder.transform(table[self.multi_value_columns]),
            columns=self.multi_value_encoder.get_feature_names_out(self.multi_value_columns),
            index = table.index
        )
        # 记录下原始 multi-value 列名 => 新的一批 multi-value 列名
        for multi_value_column in self.multi_value_columns:
            feat_info[multi_value_column]["colname_transfer"] = self.multi_value_encoder.get_feature_names_out([multi_value_column])

        table.drop(columns=self.multi_value_columns + self.category_columns, inplace=True)
        table = pd.concat([table_multi_value, table_encoded, table], axis=1, copy=False)
        # print(table.columns)
        # print(feat_info)
        return table, label, feat_info
    

class FeatureReducedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.th = 0.01
        self.features_to_keep_ = []

    def fit(self, X: Tuple[pd.DataFrame, pd.DataFrame, Dict], y=None):
        table, label, feat_info = X
        # print(feat_info)
        # 此处根据特征的方差对可能不必要的特征进行删除
        print("start to variance_threshold")
        digit_columns = []
        for col_name, col_info in feat_info.items():
            if col_info["type"] == "Number":
                digit_columns.append(col_name)
            else:
                digit_columns.extend(col_info["colname_transfer"])
        
        self.features_to_keep_ = [digit_columns[0]] # 至少有一个特征
        for column in digit_columns[1:]:  # 不能直接 table.var()，会直接内存爆掉
            if table[column].var() >= self.th:
                self.features_to_keep_.append(column)
        
        # 这里可以根据 label 对不必要的特征进行过滤
        print("end to variance_threshold fit")
        return self

    def transform(self, X: Tuple[pd.DataFrame, pd.DataFrame, Dict]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        table, label, feat_info = X
        # print(self.features_to_keep_)
        return table[self.features_to_keep_ + ["reqId"]], label, feat_info


"""
生成特征数据，用于之后平台侧进行特征重要性计算、给客户展示特征等。
fit 部分，存储详细的 feature-info, 同时存储训练时的各个特征值
transform 阶段，对 dataframe 数据进行过滤，只留取 feature_info 中配置的列
"""

class FeatureInfoSave(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        # table 中有用的列名，根据 feat-info 与当前的 table-column 交集生成
        self.col_name_in_table = []  
        # feat-info 的原始列名与特征转换后列名的映射关系
        # col_name_mapping： {
        #     "cost": "cost", # 数值型特征，没有转换
        #     "eventTime_weekday": ["eventTime_weekday_0", "eventTime_weekday_1", ...] # 非数值型特征
        # }
        self.col_name_mapping = defaultdict(list)
    
    def fit(self, X: Tuple[pd.DataFrame, pd.DataFrame, Dict], y=None):
        table, _, feat_info = X
        
        table_columns = set(table.columns)
        for feat_col_name, feat_col_info in feat_info.items():
            if feat_col_info["type"] == "Number":
                if feat_col_name in table_columns:
                    self.col_name_in_table.append(feat_col_name)
                    self.col_name_mapping[feat_col_name] = feat_col_name
            else:
                for colname_transfer in feat_col_info["colname_transfer"]:
                    if colname_transfer in table_columns:
                        self.col_name_in_table.append(colname_transfer)
                        self.col_name_mapping[feat_col_name].append(colname_transfer)

        # 从 global 中获取特征信息的保存路径
        feature_path = os.environ.get('feature_path', None)
        feature_info_path = os.environ.get('feature_info_path', None)
        if feature_info_path is None or feature_path is None:
            return self

        # 将特征信息保存下来
        # {idx: {"col_name":"", "feature_description": "", "type": "", "colname_transfer": [xxx]}}
        feature_full_info = {str(idx): {"col_name": col, **feat_info[col]}
                             for idx, col in enumerate(self.col_name_mapping.keys())}
        with open(feature_info_path, "w") as fp:
            json.dump(feature_full_info, fp, indent=2, ensure_ascii=False)

        # 存储训练时的各个特征值
        with open(feature_path+"_temp", "w") as fp:
            for _, row_value in table.iterrows():
                feat_item = f"{md5_encode(row_value['reqId'])}|"
                for feat_idx, (colname, colname_transfers) in enumerate(self.col_name_mapping.items()):
                    if feat_info[colname]["type"] == "Number":
                        feat_item += f" {feat_idx}:1:{row_value[colname]}"
                    elif feat_info[colname]["type"] == "Category":
                        for colname_transfer in colname_transfers:
                            value = md5_encode(str(row_value[colname_transfer]))
                            feat_item += f" {feat_idx}:{value}"
                    else:
                        for colname_transfer in colname_transfers:
                            sign = md5_encode(colname)
                            value = md5_encode(str(row_value[colname_transfer]))
                            feat_item += f" {feat_idx}:{sign}:{value}"
                fp.write(feat_item + "\n")
        print('save done')
        return self

    def transform(self, X) -> pd.DataFrame:
        table, _, feat_info = X
        return table[self.col_name_in_table]

