#!/bin/bash

#####AIT Dockerイメージ作成#####

DOCKER_IMAGE_NAME=ait_license_base:latest

cd "$(dirname "$0")"

# 既存削除
echo "start docker clean up..."
docker rmi "$DOCKER_IMAGE_NAME"
docker system prune -f

# ビルド
echo "start docker build..."
docker build -f ../deploy/container/dockerfile -t "$DOCKER_IMAGE_NAME" ../deploy/container

#####AITライセンス情報出力#####

LICENSE_DOCKER_IMAGE_NAME=ait_license_thirdparty_notices

cd "$(dirname "$0")"

# 既存削除
echo "start docker clean up..."
docker rmi "$LICENSE_DOCKER_IMAGE_NAME"
docker system prune -f

# ビルド
echo "start docker build..."
docker build -t "$LICENSE_DOCKER_IMAGE_NAME" -f ../deploy/container/dockerfile_license ../deploy/container

# 実行
echo "run docker..."
docker run "$LICENSE_DOCKER_IMAGE_NAME":latest > ../ThirdPartyNotices.txt

read -p "Press ENTER to continue..." answer
