#!/bin/bash

VERSION=0.1.0
TARGET_PLATFROM=x86_64-unknown-linux-gnu
# Run this script by `bash docker/build.sh` under the root dir of this repo

pwd

rm -rf docker/server
rm -rf docker/client

cargo install cross
rustup target add ${TARGET_PLATFROM}

# build server
cd custom-congestion-controller
cross build --target ${TARGET_PLATFROM} || exit 1
cd ..

# build client
cd echo
cross build --target ${TARGET_PLATFROM} || exit 1
cd ..

cp echo/target/${TARGET_PLATFROM}/debug/quic_echo_server docker/server
# cp custom-congestion-controller/target/${TARGET_PLATFROM}/debug/main docker/server
cp echo/target/${TARGET_PLATFROM}/debug/quic_echo_client docker/client

docker build -t quic-simulation:${VERSION} -f docker/Dockerfile ./docker

