#!/usr/bin/env bash

if [ $# == 0 ]; then
    echo "Supply neurontype (sphere-mainen) probetype (cylinder-fancy), coarse and box"
else
    neuron=$1
    probe=$2
    coarse=$3
    box=$4

    if [ $probe == 'cylinder' ]; then
        rad='5 10 15 20 25 30'
    fi

    for r in $rad
    do
        meshes=$(python geogen.py -neurontype $neuron -probetype $probe -boxsize $box -coarse $coarse -rad $r -dist 50 -returnfname 2>&1 >/dev/null)
        for m in $meshes
        do
            echo $m
            python ../msh_convert.py $m
        done
    done
fi
