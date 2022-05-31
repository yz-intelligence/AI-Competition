# Base Images
## 从天池基础镜像构建
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3
## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /
## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /

## 安装所需要的包
RUN pip config set global.index-url http://mirrors.aliyun.com/pypi/simple/
RUN pip config set install.trusted-host mirrors.aliyun.com
RUN pip3 install -r code/requirements.txt
RUN pip install --upgrade pip
RUN apt -y update
RUN apt install zip
RUN apt install vim -y
RUN apt install screen -y
RUN pip install catboost
RUN pip install scikit-learn
RUN pip install tqdm
RUN pip install lightgbm
RUN pip install gensim==4.1.2

## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]
