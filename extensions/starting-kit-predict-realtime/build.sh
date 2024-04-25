#!/bin/bash
set -e
#修改imgname为你自己的镜像名后，注释掉本行内容
imgname="harbor.4pd.io/lab-platform/pk_platform/model_services/ybwtest:0.1.6"
#imgname="harbor.4pd.io/lab-platform/pk_platform/model_services/automl_startingkit:0.1.1"
docker build -t $imgname .
docker push $imgname
docker rmi -f $imgname

