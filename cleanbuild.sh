#!/bin/sh
CURRENT_PATH=$(pwd)
if [ "${CURRENT_PATH:0-5}" == "build" ]
then
    cd ..
    rm -r ./build
    mkdir ./build
    cd ./build
    echo "Clean Suc"
else
    echo "Please run this script in \'build\' directory"
fi
