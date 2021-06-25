#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf

import os
import sys
from time import time
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
#For image generation and FID score
from PIL import Image
from fid_score.fid_score import FidScore

#Self import
from src.dataset_utils import load_dataset
from src.CVAE import CVAE
import src.hparams as hparams
from src.misc import Log, make_recursive_dir
import src.experiment as Experiment
from src.quality_metrics import calculate_FID
from src.analysis import analysis_correlation

#Mapping
from src.mapping.sparse import Dim_selector
from src.mapping.correlated import Correlator
from src.mapping.clustered import Cluster

parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--latent_dim", type=int, default=5) #50 for MNIST/FashionMNIST, 128 for CelebA, 5 for dsprites
parser.add_argument("--epoch", type=int, default=50)   #30 epochs for celebA
parser.add_argument("--least_epoch", type=int, default=20)   #30 epochs for celebA
parser.add_argument("--mapping", type=str, default='correlated') #['', 'sparse','clustered','correlated']
parser.add_argument("--mapping_mode", type=int, default=5) #Only For correlated: [0,1,2,3,4,5]
parser.add_argument("--mapping_submode", type=int, default=5) #Only For correlated: [0,1,2,3,4,5]
parser.add_argument("--gamma", type=float, default=0.0)
parser.add_argument("--gpu", type=str, default='7')
parser.add_argument("--dataset", type=str, default='dsprites') #['mnist', 'fashion_mnist', 'cifar10', 'celebA', 'dsprites']
parser.add_argument("--cut_by_labels_dim", type=int, default=0) #[0, 1, 2...]
parser.add_argument("--loss_type", type=str, default='cross_entropy') #['cross_entropy', 'se', 'laplace']


parser.add_argument("--binary", type=bool, default=False)
parser.add_argument("--max_nondecrease_epoch", type=int, default=5) 
parser.add_argument("--batch_size", type=int, default=100) 
parser.add_argument("--info", type=str, default='', help='add comments on experiment id')  

#FID
parser.add_argument("--FID", action='store_true', default=False)
parser.add_argument("--FID_sample_size", type=int, default=2000)

#LogP
parser.add_argument("--logP", action='store_true', default=False)
parser.add_argument("--logP_sample_size", type=int, default=2000)
parser.add_argument("--logP_particle_each", type=int, default=500)
parser.add_argument("--logP_batch_size", type=int, default=5)

#Program
parser.add_argument("--test_every", type=int, default=1)
parser.add_argument("--restore_epoch", type=int, default=0)  ##Careful
parser.add_argument("--leading_metric", type=str, default='FID')  #FID, logP, loss

#Test mode
parser.add_argument("--test_only", action='store_true', default=False)
parser.add_argument("--test_dataset_seed", type=int, default=43)


#Vamp
parser.add_argument("--vamp", action='store_true', default=False)
parser.add_argument("--vamp_input_num", type=int, default=500)
parser.add_argument("--MoG", action='store_true', default=False)
parser.add_argument("--MoG_num", type=int, default=2)

#IWAE
parser.add_argument("--IWAE", type=int, default=0)

args = parser.parse_args()

#GPU config
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
    

#id
if args.mapping=='correlated':
    experiment_id=str(args.dataset)+'_'+str(args.mapping)+'_'+str(args.latent_dim)+'_'+str(args.beta)+'_'+str(args.gamma)+'_'+str(args.mapping_mode)+'_'+str(args.mapping_submode)+'_'+args.info
else:
    experiment_id=str(args.dataset)+'_'+str(args.mapping)+'_'+str(args.latent_dim)+'_'+str(args.beta)+'_'+str(args.gamma)+'_'+args.info

if args.vamp:
    experiment_id+='_vamp_'+str(args.vamp_input_num)
if args.MoG:
    experiment_id+='_MoG_'+str(args.MoG_num)
    
#Data loading
if args.dataset=='mnist' and args.cut_by_labels_dim!=0: #Only load sample with certain label
    labels_for_mnist=list(np.arange(2**args.cut_by_labels_dim).astype(np.int32))
    experiment_id+='_cut_'+str(args.cut_by_labels_dim)
else:
    labels_for_mnist=[]

train_dataset, test_dataset, test_dataset_for_logP, image_shape, validation_dataset=load_dataset(args.dataset, args.batch_size, args.logP_batch_size, labels_for_mnist=labels_for_mnist, test_dataset_seed=args.test_dataset_seed)


#Save test samples for FID calculation
if args.FID:
    FID_sample_size=args.FID_sample_size
    test_sample_folder='FID_images_real_test/'+experiment_id+'/'
    try:
        os.system('rm -r '+test_sample_folder)
    except:
        pass
    make_recursive_dir(test_sample_folder)  
    samples_color=[]
    i=0
    for batch in test_dataset:
        samples_color.append(batch)
        i+=args.batch_size
        if i>=FID_sample_size:
            break
    samples_color=np.concatenate(samples_color, axis=0)[:FID_sample_size]
    if args.dataset in ['mnist', 'fashion_mnist']:
        samples_color=np.tile(samples_color, [1,1,1,3])    
  
    for i in range(samples_color.shape[0]):
        img=Image.fromarray((samples_color[i]*255).astype(np.uint8))
        img.save(test_sample_folder+str(i)+'.png')

if args.FID:
    FID_sample_size=args.FID_sample_size
    validation_sample_folder='FID_images_real_validation/'+experiment_id+'/'
    try:
        os.system('rm -r '+test_sample_folder)
    except:
        pass
    make_recursive_dir(test_sample_folder)  
    samples_color=[]
    i=0
    for batch in validation_dataset:
        samples_color.append(batch)
        i+=args.batch_size
        if i>=FID_sample_size:
            break
    samples_color=np.concatenate(samples_color, axis=0)[:FID_sample_size]
    if args.dataset in ['mnist', 'fashion_mnist']:
        samples_color=np.tile(samples_color, [1,1,1,3])    
  
    for i in range(samples_color.shape[0]):
        img=Image.fromarray((samples_color[i]*255).astype(np.uint8))
        img.save(test_sample_folder+str(i)+'.png')

##Load Mapping
if args.mapping=='sparse':
    with tf.device('/device:GPU:0'):
        dim_selector=Dim_selector(args.latent_dim)
elif args.mapping=='correlated':
    if args.mapping_mode==5:
    #For experiment with fixed correlation
        from src.misc import extend_matrix
        mapping_matrix=np.array([[0.5, 0],[0.5, 1]]).astype(np.float32)
        mapping_matrix=extend_matrix(mapping_matrix, args.latent_dim)
        with tf.device('/device:GPU:0'):
            correlator=Correlator(args.latent_dim, mode=args.mapping_mode, mapping_matrix=mapping_matrix)
    elif args.mapping_mode==3:
        if args.mapping_submode==0:
            split=[2,2,2,2,2,10,10,10,10]
        elif args.mapping_submode==1:
            split=[10,10,10,10,10]
        elif args.mapping_submode==2:
            split=[2,2,2,2,2]
        elif args.mapping_submode==3:
            split=[32,32,32,32]
        elif args.mapping_submode==4:
            split=[8,8,8,8,32,32,32]
        elif args.mapping_submode==5:
            split=[2,2,2,2,8,8,8,32,32,32]
        elif args.mapping_submode==6:
            split=[5,5]
        elif args.mapping_submode==7:
            split=[1,1]
        elif args.mapping_submode==8:
            split=[2,2]
        elif args.mapping_submode==9:
            split=[2,1]
        elif args.mapping_submode==10:
            split=[32,32] #64
        elif args.mapping_submode==11:
            split=[16,16,16,16] #64
        elif args.mapping_submode==12:
            split=[16,16,32] #64
        elif args.mapping_submode==13:
            split=[8,8,8,8,16,16] #64
        elif args.mapping_submode==14:
            split=[16,16] #32
        elif args.mapping_submode==15:
            split=[8,8,16] #32
        elif args.mapping_submode==16:
            split=[8,8,8,8] #32
        elif args.mapping_submode==17:
            split=[8,8] #16
        elif args.mapping_submode==18:
            split=[4,4,8] #16
        elif args.mapping_submode==19:
            split=[4,4,4,4] #16
        elif args.mapping_submode==20:
            split=[4,4] #8
        elif args.mapping_submode==21:
            split=[2,2,4] #8
        elif args.mapping_submode==22:
            split=[2,2,2,2] #8
        elif args.mapping_submode==23:
            split=[2,2] #4
        elif args.mapping_submode==24:
            split=[1,1,2] #4
        elif args.mapping_submode==25:
            split=[1,1,1,1] #4
        elif args.mapping_submode==26:
            split=[1,1] #2
            
        with tf.device('/device:GPU:0'):
            correlator=Correlator(args.latent_dim, mode=args.mapping_mode, split=split)
elif args.mapping=='clustered':
    with tf.device('/device:GPU:0'):
        cluster=Cluster(args.mapping_mode)
            

#VAE with structures from DCGAN
encoder_params, decoder_params = hparams.encoder_decoder_params(args.dataset)
with tf.device('/device:GPU:0'):
    model=CVAE(latent_dim=args.latent_dim, encoder_params=encoder_params, decoder_params=decoder_params, image_shape=image_shape)


#Experiment build
if args.mapping=='':
    #Remember to turn off 'args.gamma'
    if args.vamp:
        experiment=Experiment.Vamp_Experiment(model, args.batch_size, None, experiment_id=experiment_id, vamp_input_num=args.vamp_input_num)
    if args.MoG:
        experiment=Experiment.MoG_Experiment(model, args.batch_size, None, experiment_id=experiment_id, MoG_num=args.MoG_num)
    else:
        experiment=Experiment.Sparse_Experiment(model, args.batch_size, None, experiment_id=experiment_id)
        
elif args.mapping=='sparse':
    experiment=Experiment.Sparse_Experiment(model, args.batch_size, dim_selector, experiment_id=experiment_id)

elif args.mapping=='correlated':
    #Remember to turn off 'args.gamma'
    experiment=Experiment.Sparse_Experiment(model, args.batch_size, correlator, experiment_id=experiment_id)

elif args.mapping=='clustered':
    #Remember to turn off 'args.gamma'
    experiment=Experiment.Sparse_Experiment(model, args.batch_size, cluster, experiment_id=experiment_id)
    
#Initialize log
log_dir='log/'
make_recursive_dir(log_dir)
log_path=log_dir+experiment_id+'.txt'

            

#Begin experiment    
logP_list=[-1e7]
loss_list=[1e7]
sparse_score_list=[0.0]
FID_list=[1e7]
FID_list_validation=[1e7]

if args.restore_epoch>0:
    experiment.restore(args.restore_epoch)

initial_epoch=args.restore_epoch

##Test only mode
if args.test_only:
    #log=Log(log_path[:-6]+'_test'+log_path[-4:], append=True)
    log=Log('log/test.txt', append=True)
    experiment.restore()
    for seed in range(1):
        String=experiment_id
        
        _, test_dataset, test_dataset_for_logP, _, _=load_dataset(args.dataset, args.batch_size, args.logP_batch_size, labels_for_mnist=labels_for_mnist, test_dataset_seed=seed)
        
        
        if args.FID and seed!=0:
            FID_sample_size=args.FID_sample_size
            test_sample_folder='FID_images_real/'+experiment_id+'/'
            try:
                os.system('rm -r '+test_sample_folder)
            except:
                pass
            make_recursive_dir(test_sample_folder)  
            samples_color=[]
            i=0
            for batch in test_dataset:
                samples_color.append(batch)
                i+=args.batch_size
                if i>=FID_sample_size:
                    break
            samples_color=np.concatenate(samples_color, axis=0)[:FID_sample_size]
            if args.dataset in ['mnist', 'fashion_mnist']:
                samples_color=np.tile(samples_color, [1,1,1,3])    
  
            for i in range(samples_color.shape[0]):
                img=Image.fromarray((samples_color[i]*255).astype(np.uint8))
                img.save(test_sample_folder+str(i)+'.png')
        
        if args.FID:
            sample_folder='FID_images/'+experiment_id+'/'
            try:
                os.system('rm -r '+sample_folder)
            except:
                pass
            folder=experiment.generate_and_save_images_for_FID(args.FID_sample_size)
            print(args.FID_sample_size)
            #!FID_score=calculate_FID(folder, test_sample_folder)
            #!String+='\t'
            #!String+=str(FID_score)
            
        if args.logP:
            logP_score=experiment.logP_epoch(test_dataset_for_logP, args.logP_particle_each, args.logP_sample_size).numpy()
            String+='\t'
            String+=str(logP_score)
            
        if True:
            sparse_score=experiment.sparse_score_epoch(test_dataset)
            String+='\t'
            String+=str(sparse_score)
        log(String)
        print(String)
        print('finish seed: {}'.format(seed))
    sys.exit()

    

##Normal mode
log=Log(log_path, append=(args.restore_epoch>0))

flag=0
for epoch in range(initial_epoch+1, initial_epoch+args.epoch + 1):
    print(epoch)
    train_loss, train_ELBO, train_KL, train_reconstruction, train_time, train_info = experiment.train_epoch(train_dataset, args.beta, args.gamma, loss_type=args.loss_type, IWAE=args.IWAE)
    String='train_epoch:{} train_loss:{} train_ELBO:{} train_KL:{} train_reconstruction:{} train_time:{}'.format(epoch, train_loss, train_ELBO, train_KL, train_reconstruction, train_time)
    if np.isnan(train_loss):
        log('Enconter nan in training!')
        break
    if args.mapping=='sparse':
        String+=' sparse_loss:{}'.format(train_info[0])
    print(String)
    log(String)
    print('-'*50)
    
    if args.test_every < 0:
        if epoch%3==0:
            experiment.save(epoch)
        continue
        
    if epoch % args.test_every ==0: 
        validation_loss, _, _, _, _, _=experiment.test_epoch(validation_dataset, args.beta, args.gamma, loss_type=args.loss_type, IWAE=args.IWAE)
        test_loss, test_ELBO, test_KL, test_reconstruction, test_time, test_info=experiment.test_epoch(test_dataset, args.beta, args.gamma, loss_type=args.loss_type, IWAE=args.IWAE)
        String='test_at_epoch:{} test_loss:{} test_ELBO:{} test_KL:{} test_reconstruction:{} test_time:{}'.format(epoch, test_loss, test_ELBO, test_KL, test_reconstruction, test_time)
        loss_list.append(test_loss)
        if args.mapping=='sparse' or args.gamma>0.0:
            String+=' sparse_loss:{}'.format(test_info[0])
        sparse_score=experiment.sparse_score_epoch(test_dataset)
        String+=' sparse_score:{}'.format(sparse_score)
        sparse_score_list.append(sparse_score)
        
        if args.mapping=='correlated' and args.mapping_mode==5:
            plot_dir=log_dir+experiment_id+'/'
            make_recursive_dir(plot_dir)
            cov_mat=analysis_correlation(experiment.model, experiment.mapping, test_dataset_with_labels)
            try:
                plt.clear()
            except:
                pass
            plt.imshow(np.abs(cov_mat), cmap='gray')
            plt.colorbar()
            plt.savefig(plot_dir+str(epoch)+'.png')
        
        if args.FID:
            start_time=time()
            folder=experiment.generate_and_save_images_for_FID(args.FID_sample_size)
            FID_list.append(calculate_FID(folder, test_sample_folder))
            FID_list_validation.append(calculate_FID(folder, validation_sample_folder))
            String+=' FID:{}'.format(FID_list[-1])
            String+=' FID_time:{}'.format(time()-start_time)
        if args.logP:
            start_time=time()
            logP_list.append(experiment.logP_epoch(test_dataset_for_logP, args.logP_particle_each, args.logP_sample_size).numpy())
            String+=' logP:{}'.format(logP_list[-1])
            String+=' logP_time:{}'.format(time()-start_time)
            
        
        print(String)
        print('='*50)
        log(String)
        #Save images for quality check
        experiment.generate_and_save_images(epoch)
        experiment.reconstruct_and_save_images(epoch, test_dataset)
    
        #Decide whether to stop training
        if args.FID and args.leading_metric=='FID':
            stall_flag=(FID_list_validation[-1]>=np.min(FID_list_validation[:-1]))
        else:
            stall_flag=(loss_list[-1]>=np.min(loss_list[:-1]))
        
        if stall_flag:
            flag+=1
        else:
            flag=0
            experiment.save()
        if flag>=args.max_nondecrease_epoch and epoch>=args.least_epoch:
            log('training ends.')
            break