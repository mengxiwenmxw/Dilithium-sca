#!/usr/bin/bash
### $1: environment files name ; $2: new env name ; $3 - a dir: add a 'prefix' dir in $1 file to conda config file
conda env create -f $1 -n $2 $3
#mkdir /15T/Projects/Dilithium-SCA/scripts/env_setting/envs/
#mv ~/miniconda3/envs/$2 /15T/Projects/Dilithium-SCA/scripts/env_setting/envs/$2
conda config --add envs_dirs $3
conda activate $2