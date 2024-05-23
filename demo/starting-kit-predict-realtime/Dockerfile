# 使用基础镜像
FROM harbor.4pd.io/openmldb/openmldb:0.9.0

ADD . /root/test
WORKDIR /root/test

# 安装 Python 3.9
RUN apt-get update && \
    apt-get install -y python3.9 && \
    apt-get install -y python3-pip && \
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    ulimit -n 65535

# 设置 Python 3.9 为默认版本
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

EXPOSE 80

# 启动命令
CMD python3 main.py
