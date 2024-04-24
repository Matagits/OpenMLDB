# 使用基础镜像
FROM 4pdosc/openmldb:0.8.5

ADD . /root/test
WORKDIR /root/test

# 安装 Python 3.9
RUN apt-get update && \
    apt-get install -y python3.9 && \
    apt-get install -y python3-pip && \
    pip install -r requirements.txt

# 设置 Python 3.9 为默认版本
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

EXPOSE 80

# 启动命令
CMD python3 main.py
