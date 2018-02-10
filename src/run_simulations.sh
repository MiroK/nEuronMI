#!/usr/bin/env bash

cwd=$(pwd)

if [ $# == 0 ]; then
    echo "Supply mesh folder, (optional) boxsize (1 - 2 - 3) and coarse (0 1 2 3)"
elif [ $# == 1 ]; then
    mesh_folder=$1
    cd $mesh_folder

    files=$(ls)

    for f in $files
    do
    echo $f
    done

    cd $cwd


#    python simulate
fi