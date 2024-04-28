## 注意 ⚠️

实时榜单中，main.py 里在训练和预测中调用的是 strategy2.py，不是 strategy.py，所以可能导致实际运行的并非是修改后的代码.


## 实现我的AutoML策略

### 方法1：

实现 main.py 中的以下 x 个函数，对应竞赛场的 x 个接口。

当前代码既可以作为训练服务（只实现前两个），又可以作为预测服务（只实现后四个）。

```python
def post_init_data():
    pass

def post_append_data():
    pass

def get_status(task_id):
    pass

def load_model():
    pass

def load_status(task_id):
    pass

def predict_stage():
    pass
```

具体输入输出参考竞赛榜的提交方法介绍

### 方法2：

实现 strategy.py 中的以下三个函数，实现训练（包含特征工程、调参、训练等等）、加载模型（包含加载模型、数据等内容）和预测功能：

```python
def train():
    pass

def load_model_and_data():
    pass

def predict():
    pass
```

具体输入输出参考 strategy.py 中的注释。如果有不清楚，范式+ 联系 @王世超。

## 关于当前代码的说明

通过 sklearn 的 pipeline 进行训练和预测，里面的代码可根据需要自定义调整。

**pipeline.py 的说明**

- FeatureEngineerInitTransformer: 抽取数值型特征、类别型特征、多值型特征
- FeatureInfoSave: 将特征数据保存下来，后续大概率会用于给客户展示
- FeatureEngineerTransformer: 将类别型特征、多值型特征进行 one-hot 形式展开，从而可以计算
- LogisticRegression: 逻辑回归拟合数据

**pipeline2.py 的说明**

- FeatureEngineerInitTransformer: 抽取数值型特征、类别型特征、多值型特征
- FeatureEngineerTransformer: 将类别型特征、多值型特征进行 one-hot 形式展开，从而可以计算
- FeatureReducedTransformer: 根据各个特征的方差来减少特征量
- FeatureInfoSave: 将特征数据保存下来，后续大概率会用于给客户展示
- LogisticRegression: 逻辑回归拟合数据

也可以不选择使用 pipeline 的形式进行训练和预测。

## 生成镜像

将 build.sh 中 imgname 修改为你自己的镜像名，执行

```shell
bash ./build.sh
```

## 提交

参考竞赛帮提交方法介绍，提交策略。一个参考的策略配置如下：

若不配置资源，默认容器为 0.4 核 CPU、500M 内存。

```yaml
train:
  docker_image: harbor.4pd.io/xxxx:tag
  values:
    env: # 配置环境变量，若不需要直接删掉
      - name: env_test
        value: hello
      - name: env_test2
        value: hello2
    resources: # 配置资源大小
      limits:
        cpu: 4000m
        memory: 10240Mi
predict:
  docker_image: harbor.4pd.io/xxxx:tag
  values:
    env: # 配置环境变量，若不需要直接删掉
      - name: env_test
        value: hello
      - name: env_test2
        value: hello2
    resources: # 配置资源大小
      limits:
        cpu: 4000m
        memory: 10240Mi
other: # 其他属性，不需要直接删掉
  concurrency: 3  # v0 版本默认 3 并发，测评时最低 3 并发
```

## 如何进行单机调试

可以使用 sample 数据进行 strategy 调试：

### 训练过程

```bash
workspace_path=一个文件路径，table-schema、task-type、train-data、model 都会写入到该文件夹下，由后续阶段加载
feat_path=hi.txt 
feat_out_path=hi.json
python strategy.py train anta-sample/table_schema.json BinaryClassification ${workspace_path} ${feat_path} ${feat_out_path} anta-sample/0000
```

### 加载与预测过程

```bash
workspace_path=一个文件路径，table-schema、task-type、train-data、model 都会写入到该文件夹下，由后续阶段加载
python strategy.py load_and_predict ${workspace_path} anta-sample/0000/flattenRequest 
```

