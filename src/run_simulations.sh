#!/usr/bin/env bash

cwd=$(pwd)

if [ $# == 0 ]; then
    echo "Supply mesh folder (includeing final /), (optional) coarse (0 - 1 - 2 - 3) and boxsize (1 2 3)"
elif [ $# == 1 ]; then
    mesh_folder=$1

    files=$(ls $mesh_folder)

    for f in $files
    do
    echo "Simulating: "$f"_wprobe.h5"
    python simulate_emi.py -mesh "$mesh_folder$f/$f"_wprobe.h5
    echo "Simulating: "$f"_noprobe.h5"
    python simulate_emi.py -mesh "$mesh_folder$f/$f"_noprobe.h5 -probemesh "$mesh_folder$f/$f"_wprobe.h5
    done
elif [ $# == 2 ]; then
    mesh_folder=$1
    coarse=$2
    files=$(ls $mesh_folder)

    for f in $files
    do
    if [[ "$f" == *"coarse_$coarse"* ]]; then
        echo "Simulating: "$f"_wprobe.h5"
        python simulate_emi.py -mesh "$mesh_folder$f/$f"_wprobe.h5
        echo "Simulating: "$f"_noprobe.h5"
        python simulate_emi.py -mesh "$mesh_folder$f/$f"_noprobe.h5 -probemesh "$mesh_folder$f/$f"_wprobe.h5
    fi
    done
elif [ $# == 3 ]; then
    mesh_folder=$1
    coarse=$2
    box=$3
    files=$(ls $mesh_folder)

    for f in $files
    do
    if [[ "$f" == *"coarse_$coarse"* ]] && [[ "$f" == *"box_$box"* ]]; then
        echo "Simulating: "$f"_wprobe.h5"
        python simulate_emi.py -mesh "$mesh_folder$f/$f"_wprobe.h5
        echo "Simulating: "$f"_noprobe.h5"
        python simulate_emi.py -mesh "$mesh_folder$f/$f"_noprobe.h5 -probemesh "$mesh_folder$f/$f"_wprobe.h5
    fi
    done
fi
