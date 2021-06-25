#!/bin/bash
gpus=(0) #GPU sharing and multi-GPU
betas=(1.0)
datasets=('mnist')
gammas=(1e-7 0 0.3 3.0 0.1 10.0 0.03 30.0 0.01 100)
infos=(0 1 2 3 4 5 6 7 8 9) #Repeat experiments 
latent_dims=(50)


#echo ${#betas[@]}
gpu_stat=-1
for ((q=0;q<${#datasets[@]};q++)); do
    for ((p=0;p<${#infos[@]};p++)); do
        for (( i=0;i<${#betas[@]};i++)); do
            for (( j=0;j<${#gammas[@]};j++)); do
                for ((k=0;k<${#latent_dims[@]};k++)); do
            
                    let gpu_stat=gpu_stat+1
        
                    if [ $gpu_stat -eq ${#gpus[@]} ]; then
                        wait
                        let gpu_stat=0
                    fi
            
                    python inteL_VAE.py --gpu=${gpus[$gpu_stat]} --epoch=200 --least_epoch=20 --beta=${betas[$i]} --latent_dim=${latent_dims[$k]} --gamma=${gammas[$j]} --dataset=${datasets[$q]} --binary=False --mapping=''  --info=${infos[$p]} --FID --FID_sample_size=2000 --IWAE=50  &
                    
                done
            done
        done 
    done
done
