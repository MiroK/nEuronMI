#!/usr/bin/env bash

if [ $# == 0 ]; then
    echo "Supply neurontype (sphere-mainen) probetype (cylinder-fancy), coarse and box"
else
    neuron=$1
    probe=$2
    coarse=$3
    box=$4

    if [ $probe == 'fancy' ]; then
        #tip='40,60,-100 40,50,-100 40,40,-100 40,30,-100 40,20,-100 40,10,-100'
	tip='40,100,-100 40,80,-100 40,60,-100 40,50,-100 40,40,-100 40,30,-100 40,20,-100 40,10,-100'
    elif [ $probe == 'cylinder' ]; then
        tip='"40,40,-100" "40,30,-100" "40,20,-100" "40,10,-100"'
    fi

    for t in $tip
    do
        echo $t
        meshes=$(python geogen.py -neurontype $neuron -probetype $probe -boxsize $box -coarse $coarse -probetip $t -returnfname 2>&1 >/dev/null)
        for m in $meshes
        do
            echo $m
            python ../msh_convert.py $m
        done
    done
fi
