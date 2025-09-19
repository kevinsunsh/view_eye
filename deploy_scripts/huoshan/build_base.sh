#!/bin/bash

docker build --platform=linux/amd64 --no-cache -t mengs-cn-beijing.cr.volces.com/aura/view_eye:basev1 -f ../../src/Dockerfile_base ../../src

# 推送 Docker 镜像
docker push mengs-cn-beijing.cr.volces.com/aura/view_eye:basev1
