#!/usr/bin/env bash

if [ $# == 0 ]; then
    echo "Supply neurontype (sphere-mainen) probetype (cylinder-fancy), coarse and box"
else
    neuron=$1
    probe=$2
    coarse=$3
    box=$4

    if [ $probe == 'fancy' ]; then
	dist=70
	rot='30 60 90 120 150 180'
    fi

    for r in $rot
    do
        echo $r
        meshes=$(python geogen.py -neurontype $neuron -probetype $probe -boxsize $box -coarse $coarse -dist $dist -rot $r -returnfname 2>&1 >/dev/null)
        for m in $meshes
        do
            echo $m
            python ../msh_convert.py $m
        done
    done
fi
