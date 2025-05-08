#!/bin/bash

BUCKET=https://dl.fbaipublicfiles.com/ego-topo

function dls3 {
    mkdir -p `dirname $1`
    wget -O $1 $BUCKET/$1
}

function dls3_unzip {
    mkdir -p `dirname $1`
    wget -O $1 $BUCKET/$1
    unzip $1 -d `dirname $1`
    rm $1
}

# download dataset metadata -- train/val splits etc. (628K + 238K)
# Required for all experiments
dls3_unzip data/epic/splits.zip
dls3_unzip data/gtea/splits.zip
echo 'Downloaded dataset splits' 