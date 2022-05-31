# 第三届阿里云磐久智维算法大赛亚军方案

## 官网地址

https://tianchi.aliyun.com/competition/entrance/531947/introduction

## 项目目录结构
```
├── Dockerfile 
├── README.md
├── code
│   ├── catboost_fs.py +++++++++++++++++++++++++++++++ 模型训练代码
│   ├── generate_feature.py ++++++++++++++++++++++++++ 特征生成代码
│   ├── generate_pseudo_label.py  ++++++++++++++++++++ 伪标签代码
│   ├── get_crashdump_venus_fea.py +++++++++++++++++++ 新数据特征生成代码
│   ├── requirements.txt +++++++++++++++++++++++++++++ python包版本
│   ├── stacking.py ++++++++++++++++++++++++++++++++++ 模型融合代码
│   └── utils.py +++++++++++++++++++++++++++++++++++++ 小工具脚本
├── data
│   ├── preliminary_a_test +++++++++++++++++++++++++++ 初赛A榜测试数据集
│   ├── preliminary_b_test +++++++++++++++++++++++++++ 初赛B榜测试数据集
│   └── preliminary_train ++++++++++++++++++++++++++++ 训练集数据
├── docker_push.sh +++++++++++++++++++++++++++++++++++++++++ Docker镜像构建、push脚本
├── feature
│   └── generation +++++++++++++++++++++++++++++++++++ 特征生成文件夹
├── log ++++++++++++++++++++++++++++++++++++++++++++++++++++ 日志文件夹
│   ├── catboost.log +++++++++++++++++++++++++++++++++ 模型运行日志
├── model ++++++++++++++++++++++++++++++++++++++++++++++++++ 模型文件
├── prediction_result ++++++++++++++++++++++++++++++++++++++ 模型预测结果文件夹
│   ├── cat_prob_result.csv ++++++++++++++++++++++++++ CATBOOST模型预测概率
│   ├── catboost_result.csv ++++++++++++++++++++++++++ CATBOOST模型预测结果
│   └── stacking_result.csv ++++++++++++++++++++++++++ 模型融合结果
├── run.log ++++++++++++++++++++++++++++++++++++++++++++++++ 代码运行日志
├── run.sh  ++++++++++++++++++++++++++++++++++++++++++++++++ 代码运行脚本
├── tcdata  ++++++++++++++++++++++++++++++++++++++++++++++++ 复赛测试集数据文件夹(具体文件请使用初赛相关文件更改文件名替换)
│   ├── final_crashdump_dataset_b.csv ++++++++++++++++ 复赛B榜新数据文件
│   ├── final_sel_log_dataset_b.csv ++++++++++++++++++ 复赛测试集日志文件
│   ├── final_submit_dataset_b.csv +++++++++++++++++++ 复赛测试集ID
│   └── final_venus_dataset_b.csv ++++++++++++++++++++ 复赛B榜新数据文件
├── user_data
│   └── tmp_data +++++++++++++++++++++++++++++++++++++ 临时文件
└── 答辩PPT
    └── 悦智AI实验室_20220525.pdf
```
## 运行环境
Python版本为3.8，各个Python包版本见requirements.txt，使用如下命令即可安装：
```
pip install -r code/requirements.txt
```

## 构建镜像运行代码
### 构建镜像
```
docker build -t [你的镜像仓库]:[TAG] .
```
### 运行镜像
```
docker run  [你的镜像ID] sh run.sh 
```
### push 镜像
```
docker push [你的仓库地址]:[TAG]
```
### 运行&push 镜像
```
bash docker_push.sh
```

## 运行代码
```
bash run.sh
```