#!/usr/bin/env bash

if [ $# == 0 ]; then
    echo "Supply neurontype (sphere-mainen) and probetype (cylinder-fancy), and (optional) coarse and box size"
elif [ $# == 2 ]; then
    neuron=$1
    probe=$2
    box='1 2 3 4 5'
    coarse='0 1 2 3'

    for b in $box
        do
        for c in $coarse
            do
                meshes=$(python geogen.py -neurontype $neuron -probetype $probe -boxsize $b -coarse $c -returnfname 2>&1 >/dev/null)
                for m in $meshes
                    do
                        echo $m
                        python ../msh_convert.py $m
                    done
            done
        done
elif [ $# == 4 ]; then
    neuron=$1
    probe=$2
    coarse=$3
    box=$4

    meshes=$(python geogen.py -neurontype $neuron -probetype $probe -boxsize $box -coarse $coarse -returnfname 2>&1 >/dev/null)
    for m in $meshes
        do
            echo $m
            python ../msh_convert.py $m
        done
fi
