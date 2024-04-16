FROM harbor.4pd.io/lab-platform/inf/python:3.9
ADD . /root/test
WORKDIR /root/test 
EXPOSE 80
RUN pip install -r requirements.txt -i https://nexus.4pd.io/repository/pypi-all/simple
CMD python main.py 

