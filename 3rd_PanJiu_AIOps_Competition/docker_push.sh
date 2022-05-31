# 创建镜像 并提交到你的镜像仓库
rm -rf result.zip
# built 镜像
docker build -t [你的仓库地址]:[TAG] .
# push 镜像
docker push [你的仓库地址]:[TAG]