#!/usr/bin/env bash

if [ $# == 0 ]; then
    echo "Supply neurontype (sphere-mainen) and probetype (cylinder-fancy)"
else
    neuron=$1
    probe=$2
    box='1 2 3'
    coarse='0 1 2 3'

    for b in $box
        do
        for c in $coarse
            do
                meshes=$(python geogen.py -neurontype $neuron -probetype $probe -box $b -coarse $c -returnfname 2>&1 >/dev/null)
                for m in $meshes
                    do
                        echo $m
                        python ../msh_convert.py $m
                    done
            done
        done
fi