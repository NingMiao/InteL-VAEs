#!/bin/bash
gpus=(0) #GPU sharing and multi-GPU
betas=(1.0)
latent_dims=(128 64 32 16 8 4 2)
infos=(0 1 2 3 4 5 6 7 8 9) #Repeat experiments

gpu_stat=-1
for ((i=0;i<${#infos[@]};i++)); do
    for ((k=0; k<${#latent_dims[@]};k++)); do
        if [ ${latent_dims[$k]} -eq 64 ]; then
            mapping_submodes=(10 11 12 13)
        fi
        if [ ${latent_dims[$k]} -eq 32 ]; then
            mapping_submodes=(14 16)
        fi
        if [ ${latent_dims[$k]} -eq 16 ]; then
            mapping_submodes=(17 18 19)
        fi
        if [ ${latent_dims[$k]} -eq 8 ]; then
            mapping_submodes=(20 21 22)
        fi
        if [ ${latent_dims[$k]} -eq 4 ]; then
            mapping_submodes=(23 24 25)
        fi
        if [ ${latent_dims[$k]} -eq 2 ]; then
            mapping_submodes=(26)
        fi

        for ((j=0;j<${#mapping_submodes[@]};j++)); do    
            #echo $j, $k, ${#mapping_submodes[$j]}, ${latent_dims[$k]}
            let gpu_stat=gpu_stat+1
            if [ $gpu_stat -eq ${#gpus[@]} ]; then
                wait
                let gpu_stat=0
            fi
            if [ ${infos[$i]} -eq -1 ]; then
                python inteL_VAE.py --gpu=${gpus[$gpu_stat]} --epoch=100 --beta=1.0 --latent_dim=2 --dataset='fashion_mnist' --mapping=''&
            else
                python inteL_VAE.py --gpu=${gpus[$gpu_stat]} --epoch=100 --least_epoch=20 --beta=1.0 --latent_dim=${latent_dims[$k]} --dataset='celebA' --mapping='correlated' --mapping_mode=3 --mapping_submode=${mapping_submodes[$j]} --info=${infos[$i]}&
            fi
            
        done
    done
done