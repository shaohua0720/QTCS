#!/bin/bash
# python main_res_vae_quant_Train.py --network "NDVitResVAE" --num_embeddings 4
# python main_res_vae_quant_Train.py --network "NDVitResVAE" --num_embeddings 16
# python main_res_vae_quant_Train.py --network "NDVitResVAE" --num_embeddings 32
# python main_res_vae_quant_Train.py --network "NDVitResVAE" --num_embeddings 64
# python main_res_vae_quant_Train.py --network "NDVitResVAE" --num_embeddings 64 --nd_enable 0

# python main_res_vae_quant_Train.py --network "NDVitAE" --succ_prob 0.001953125
# python main_res_vae_quant_Train.py --network "NDVitAE" --succ_prob 0.0039
# python main_res_vae_quant_Train.py --network "NDVitAE" --nd_enable 0
# python main_res_vae_quant_Train.py --network "SLViTAE" --h1h2h3 16 16 16 --hidden_dim 64
# python main_res_vae_quant_Train.py --network "SLViTAE" --h1h2h3 16 32 4 --hidden_dim 128
# python main_res_vae_quant_Train.py --network "SLViTAEQuant" --num_embeddings 4
# python main_res_vae_quant_Train.py --network "SLViTAEQuant" --num_embeddings 16
# python main_res_vae_quant_Train.py --network "SLViTAEQuant" --num_embeddings 32
# python main_res_vae_quant_Train.py --network "SLViTAEQuant" --num_embeddings 64

# for distributed running
# torchrun --standalone --nproc_per_node=4 multigpu_torchrun.py --total_epochs 200 --save_every 10
# torchrun --standalone --nproc_per_node=4 multigpu_torchrun.py

# train='/home/shaohua/Documents/datasets/qcsmimo/qdg_umi5g_3p84/umi_train.h5'
# val='/home/shaohua/Documents/datasets/qcsmimo/qdg_umi5g_3p84/umi_val.h5'

train='/home/shaohua/Documents/datasets/qcsmimo_rev1/3GPP_38.901_UMa_NLOS_train.h5'
val='/home/shaohua/Documents/datasets/qcsmimo_rev1/3GPP_38.901_UMa_NLOS_val.h5'

args="--train_data ${train} --val_data ${val} \
    --test_data ${val} --device cuda:1 --ratio 0.25"
python Train.py ${args} 1>log.txt 2>&1
