#! /bin/bash

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
DATA_DIR=${SCRIPTPATH}/dataset_shapenet


echo "### Installing Python packages ###"
python3 -m pip install -r ${SCRIPTPATH}/requirements.txt


echo "\n### Downloading the ShapeNet dataset ###"
if [ -d $DATA_DIR ]; then
    echo "$DATA_DIR already exists, if you haven't downloaded the dataset, please remove this folder and try again"
else
    if ! [ -x "$(command -v unzip)" ]; then
        echo "'unzip' could not be found, please install it with 'sudo apt install -y 'unzip'" >&2
        exit 1
    fi
    wget -nc https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip --no-check-certificate
    echo "Creating folder $DATA_DIR and unzipping the dataset"
    mkdir $DATA_DIR && unzip shapenetcore_partanno_segmentation_benchmark_v0.zip -d $DATA_DIR && rm shapenetcore_partanno_segmentation_benchmark_v0.zip
fi


DATA_DIR=${SCRIPTPATH}/dataset_novel
if ! [ -d $DATA_DIR ]; then
    echo "\n### Creating the folder for the Novel Categories dataset ###"
    mkdir $DATA_DIR
fi