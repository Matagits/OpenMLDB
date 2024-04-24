#!/bin/bash
set -e
#修改imgname为你自己的镜像名后，注释掉本行内容
imgname="demo.test/automl:0.0.1"
docker rm -f automl
docker rmi -f $imgname
docker build -t $imgname .
docker run -it --name automl -p 8090:80 $imgname
