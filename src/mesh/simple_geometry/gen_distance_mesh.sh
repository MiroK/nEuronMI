#!/usr/bin/env bash

if [ $# == 0 ]; then
    echo "Supply neurontype (sphere-mainen) probetype (cylinder-fancy), coarse and box"
else
    neuron=$1
    probe=$2
    coarse=$3
    box=$4

    if [ $probe == 'fancy' ]; then
        dist='25 30 35 45 55 85'
    elif [ $probe == 'cylinder' ]; then
        dist='32.5 37.5 42.5 52.5 62.5 92.5'
    elif [ $probe == 'pixel' ]; then
        dist='27.5 32.5 37.5 47.5 57.5 87.5'
    fi

    for d in $dist
    do
        meshes=$(python geogen.py -neurontype $neuron -probetype $probe -boxsize $box -coarse $coarse -dist $d -returnfname 2>&1 >/dev/null)
        for m in $meshes
        do
            echo $m
            python ../msh_convert.py $m
        done
    done
fi
