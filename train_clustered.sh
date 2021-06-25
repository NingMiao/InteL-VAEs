#!/bin/bash
gpus=(0 0 0 0) #GPU sharing and multi-GPU
betas=(1.0)
infos=(501 502 503 504 505 506 507 508 509 510 511 512 513 514 515) #Repeat experiments 
mappings=(vamp vanilla clustered MoG)
cuts=(1)
least_epoch=30
epoch=60
gamma=0.0

latent_dim=1
gpu_stat=-1

for ((i=0;i<${#infos[@]};i++)); do
    for ((j=0;j<${#mappings[@]};j++)); do
        for ((k=0;k<${#cuts[@]};k++)); do
            let gpu_stat=gpu_stat+1
            if [ $gpu_stat -eq ${#gpus[@]} ]; then
                wait
                echo wait
                let gpu_stat=0
            fi
           
            if [ ${mappings[$j]} = 'vanilla' ]; then
                python inteL_VAE.py --gpu=${gpus[$gpu_stat]} --epoch=$epoch --least_epoch=$least_epoch --beta=1.0 --gamma=$gamma --latent_dim=$latent_dim --dataset='mnist' --mapping='' --info=${infos[$i]} --FID --leading_metric='FID' --cut_by_labels_dim=${cuts[$k]} --test_only &
            elif [ ${mappings[$j]} = 'vamp' ]; then
                python inteL_VAE.py --gpu=${gpus[$gpu_stat]} --epoch=$epoch --least_epoch=$least_epoch --beta=1.0 --gamma=$gamma --latent_dim=$latent_dim --dataset='mnist' --mapping='' --info=${infos[$i]} --FID --leading_metric='FID' --cut_by_labels_dim=${cuts[$k]} --vamp  --test_only &
            elif [ ${mappings[$j]} = 'MoG' ]; then
                let MoG_num=2**${cuts[$k]}
                python inteL_VAE.py --gpu=${gpus[$gpu_stat]} --epoch=$epoch --least_epoch=$least_epoch --beta=1.0 --gamma=$gamma --latent_dim=$latent_dim --dataset='mnist' --mapping='' --info=${infos[$i]} --FID --leading_metric='FID' --cut_by_labels_dim=${cuts[$k]} --MoG --MoG_num=$MoG_num --test_only &
            else
                python inteL_VAE.py --gpu=${gpus[$gpu_stat]} --epoch=$epoch --least_epoch=$least_epoch --beta=1.0 --gamma=$gamma --latent_dim=$latent_dim --dataset='mnist' --mapping='clustered' --mapping_mode=${cuts[$k]} --info=${infos[$i]} --FID --leading_metric='FID' --cut_by_labels_dim=${cuts[$k]} --test_only &
            fi
        done
    done
done
