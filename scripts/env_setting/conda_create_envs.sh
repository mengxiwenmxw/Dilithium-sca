#!/usr/bin/bash
####################################################################
### Use this scripts when conda version dont surport --prefix with --name 
### More recommend to use: conda env create -f env.yml -n env_name
###
### $1: environment files name ; 
### $2: new env name ; 
### $3 - a dir: add a 'prefix' dir in $1 file to conda config file;
### Must install conda pack use: conda install -n base conda-pack
####################################################################
echo "--> Start to create env $2 at root envs dir:"
conda env create -f $1 -n $2 
echo "--> Start to pack and mv env $2 to dir $3 :"
conda pack -n $2 -o $2.tar.gz
ENV_PATH=$3/$2
if [ -d $ENV_PATH ];then
echo "INFO: Directory $ENV_PATH exists. skip to mkdir."
else 
mkdir -p $ENV_PATH
fi
echo "--> Unzip env ..."
tar -xzf $2.tar.gz -C $ENV_PATH
echo "--> Remove $2.tar.gz"
rm -f $2.tar.gz
echo "--> Add $3 to conda envs_dirs"
conda config --add envs_dirs $3
echo "--> Activate $2"
conda activate $2