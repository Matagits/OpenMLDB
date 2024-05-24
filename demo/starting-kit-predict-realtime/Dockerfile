FROM harbor.4pd.io/lab-platform/pk_platform/model_services/ybwtest:1.0.0
RUN rm -rf /root/test

ADD . /root/test
WORKDIR /root/test

EXPOSE 80

# 启动命令
CMD python3 main.py
