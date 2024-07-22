#!/bin/bash

VERSION=0.1.0

# Detect the operating system
os_name=$(uname -s)

os_var=""

# Set os_var based on the detected operating system
case "$os_name" in
    Linux)
        os_var="linux"
        BUILD_TOOL=cargo
        TARGET_PLATFROM=""
        BUILD_FLAG=""
        TARGET_PLATFROM_DIR=""
        ;;
    Darwin)
        os_var="macos"
        cargo install cross
        TARGET_PLATFROM=x86_64-unknown-linux-gnu
        rustup target add ${TARGET_PLATFROM}
        BUILD_FLAG="--target ${TARGET_PLATFROM}"
        TARGET_PLATFROM_DIR=${TARGET_PLATFROM}/
        BUILD_TOOL=cross
        ;;
    *)
        os_var="unknown"
        ;;
esac

pwd

rm -rf docker/server
rm -rf docker/client

# build server
cd custom-congestion-controller
${BUILD_TOOL} build ${BUILD_FLAG} || exit 1
cd ..

# build client
cd echo
${BUILD_TOOL} build ${BUILD_FLAG} || exit 1
cd ..

cp echo/target/${TARGET_PLATFROM_DIR}debug/quic_echo_server docker/server
# cp custom-congestion-controller/target/${TARGET_PLATFROM_DIR}debug/main docker/server
cp echo/target/${TARGET_PLATFROM_DIR}debug/quic_echo_client docker/client

docker build -t quic-simulation:${VERSION} -f docker/Dockerfile ./docker

