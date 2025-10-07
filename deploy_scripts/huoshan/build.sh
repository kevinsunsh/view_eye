#!/bin/bash

# 设置新版本号
NEW_VERSION="0.0.39"

# 尝试使用 Docker 命令
echo "构建 Docker 镜像..."
if docker build --platform=linux/amd64 --no-cache -t mengs-cn-beijing.cr.volces.com/aura/view_eye:${NEW_VERSION} -f ../../src/Dockerfile ../../src; then
    echo "Docker 镜像构建成功"
else
    echo "错误: Docker 镜像构建失败"
    exit 1
fi

# 推送 Docker 镜像
echo "推送 Docker 镜像..."
if docker push mengs-cn-beijing.cr.volces.com/aura/view_eye:${NEW_VERSION}; then
    echo "Docker 镜像推送成功"
else
    echo "错误: Docker 镜像推送失败"
    exit 1
fi

# 更新 s.yaml 中的版本号
echo "更新版本号..."
if sed -i '' "s/version: \"[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\"/version: \"${NEW_VERSION}\"/" s.yaml; then
    echo "版本号更新成功"
else
    echo "错误: 版本号更新失败"
    exit 1
fi

# 部署
echo "开始部署..."
s deploy
