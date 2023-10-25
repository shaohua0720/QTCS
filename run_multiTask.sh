#!/bin/bash
para=(1 125 25 5) 
device=(0 1 2 3)
log=(1 125 25 5)
ratio=(0.1 0.125 0.25 0.5)
train='/home/shaohua/Documents/datasets/qcsmimo/qdg_uma5g_3p84/uma_train.h5'
val='/home/shaohua/Documents/datasets/qcsmimo/qdg_uma5g_3p84/uma_val.h5'
pwd=`pwd`
for i in $(seq 0 3); do
    name=qcsmimo0p${para[i]}_8bit
    dev=cuda:${device[i]}
    rm -rf ${pwd}/${name}/results
    rm -rf ${pwd}/${name}/"log_${log[i]}.txt"
    cd ${pwd}/${name}
    # echo `pwd`
    args="--train_data ${train} --val_data ${val} \
    --test_data ${val} --device ${dev} --ratio ${ratio[i]}"
    python Train.py ${args} 1>log_${log[i]}.txt 2>&1 &
    cd ${pwd}
done