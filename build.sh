#!/usr/bin/sh

if [ "$2" = "rebuild" ]; then
    rebuild_cmd="--fresh"
else
    rebuild_cmd=""
fi

# if debug flag is set, build in debug mode
if [ "$1" = "debug" ]; then
    cmake -G Ninja -B build-debug -DCMAKE_BUILD_TYPE=Debug -DGPU=ON ${rebuild_cmd}
    cmake --build build-debug
    exit 0
else 
    cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DGPU=ON ${rebuild_cmd}
    cmake --build build
fi