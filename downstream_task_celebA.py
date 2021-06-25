import tensorflow as tf

import os
from time import time
from copy import deepcopy as copy
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
from src.experiment import reparameterize
from src.quality_metrics import calculate_FID
from src.analysis import analysis_correlation
from src.analysis import classifier_experiment

#Mapping
from src.mapping.sparse import Dim_selector
from src.mapping.correlated import Correlator
from src.mapping.clustered import Cluster
from src.analysis import cov


parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--latent_dim", type=int, default=128) #50 for MNIST/FashionMNIST, 128 for CelebA, 5 for dsprites
parser.add_argument("--epoch", type=int, default=50)   #30 epochs for celebA
parser.add_argument("--least_epoch", type=int, default=20)   #30 epochs for celebA
parser.add_argument("--mapping", type=str, default='sparse') #['', 'sparse','clustered','correlated']
parser.add_argument("--mapping_mode", type=int, default=3) #For correlated: [0,1,2,3,4,5]
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--gpu", type=str, default='')
parser.add_argument("--dataset", type=str, default='celebA') #['mnist', 'fashion_mnist', 'cifar10', 'celebA', 'dsprites']
parser.add_argument("--loss_type", type=str, default='cross_entropy') #['cross_entropy', 'se']

parser.add_argument("--binary", type=bool, default=False)
parser.add_argument("--max_nondecrease_epoch", type=int, default=5) 
parser.add_argument("--batch_size", type=int, default=100) 
parser.add_argument("--info", type=str, default='0', help='add comments on experiment id')  

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

#For distributed training
parser.add_argument("--feature_start_id", type=int, default=0)
parser.add_argument("--feature_end_id", type=int, default=10)

args = parser.parse_args()

print(args.feature_start_id, args.feature_end_id)
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
    
#Data loading
test_dataset_with_labels, image_shape, label_names=load_dataset(args.dataset, args.batch_size, analysis_mode=True)
channel=image_shape[-1]

encoder_params, decoder_params = hparams.encoder_decoder_params(args.dataset)

info=int(args.info)
experiments_0=[['correlated', 14, [info]], ['correlated', 15, [info]], ['correlated', 16, [info]]]
experiments_1=[['correlated', 17, [info]], ['correlated', 18, [info]], ['correlated', 19, [info]]]
experiments_2=[['correlated', 20, [info]], ['correlated', 21, [info]], ['correlated', 22, [info]]]
experiments_3=[['correlated', 24, [info]]]
experiments_4=[['correlated', 26, [info]]]
experiments=experiments_0+experiments_1+experiments_2+experiments_3+experiments_4

def get_feature(experiment_setting):
    mapping=experiment_setting[0]
    info=experiment_setting[2]
    #id
    if experiment_setting[0] == 'correlated':
        mapping_mode=3
        latent_dim=128
        gamma=args.gamma
        if experiment_setting[1] in [10,11,12,13]:
            latent_dim=64
            gamma=0.0
        if experiment_setting[1] in [14, 15, 16]:
            latent_dim=32   
            gamma=0.0
        if experiment_setting[1] in [17, 18, 19]:
            latent_dim=16   
            gamma=0.0
        if experiment_setting[1] in [20, 21, 22]:
            latent_dim=8 
            gamma=0.0
        if experiment_setting[1] in [23, 24, 25]:
            latent_dim=4
            gamma=0.0
        if experiment_setting[1] in [26]:
            latent_dim=2 
            gamma=0.0
        mapping_submode=str(experiment_setting[1])
        experiment_id=str(args.dataset)+'_'+str(mapping)+'_'+str(latent_dim)+'_'+str(args.beta)+'_'+str(gamma)+'_'+str(mapping_mode)+'_'+str(mapping_submode)+'_'+str(int(info))
    elif str.isdigit(experiment_setting[0]):
        mapping=''
        gamma=0.0
        latent_dim=int(experiment_setting[0])
        experiment_id=str(args.dataset)+'_'+str(mapping)+'_'+str(latent_dim)+'_'+str(args.beta)+'_'+str(gamma)+'_'+str(int(info))
    else:
        latent_dim=128
        gamma=experiment_setting[1]
        experiment_id=str(args.dataset)+'_'+str(mapping)+'_'+str(latent_dim)+'_'+str(args.beta)+'_'+str(gamma)+'_'+str(int(info))
       
    ##Load Mapping
    if mapping=='sparse':
        with tf.device('/device:GPU:0'):
            dim_selector=Dim_selector(args.latent_dim)
    elif mapping=='correlated':
        if mapping_mode==5:
        #For experiment with fixed correlation
            from src.misc import extend_matrix
            mapping_matrix=np.array([[0.5, 0],[0.5, 1]]).astype(np.float32)
            mapping_matrix=extend_matrix(mapping_matrix, args.latent_dim)
            with tf.device('/device:GPU:0'):
                correlator=Correlator(args.latent_dim, mode=mapping_mode, mapping_matrix=mapping_matrix)
        elif mapping_mode==3:
            if mapping_submode=='0':
                split=[2,2,2,2,2,10,10,10,10]
            elif mapping_submode=='1':
                split=[10,10,10,10,10]
            elif mapping_submode=='2':
                split=[2,2,2,2,2]
            elif mapping_submode=='3':
                split=[32,32,32,32]
            elif mapping_submode=='4':
                split=[8,8,8,8,32,32,32]
            elif mapping_submode=='5':
                split=[2,2,2,2,8,8,8,32,32,32]
            elif mapping_submode=='10':
                split=[32,32] #64
            elif mapping_submode=='11':
                split=[16,16,16,16] #64
            elif mapping_submode=='12':
                split=[16,16,32] #64
            elif mapping_submode=='13':
                split=[8,8,8,8,16,16] #64
            elif mapping_submode=='14':
                split=[16,16] #32
            elif mapping_submode=='15':
                split=[8,8,16] #32
            elif mapping_submode=='16':
                split=[8,8,8,8] #32
            elif mapping_submode=='17':
                split=[8,8] #16
            elif mapping_submode=='18':
                split=[4,4,8] #16
            elif mapping_submode=='19':
                split=[4,4,4,4] #16
            elif mapping_submode=='20':
                split=[4,4] #8
            elif mapping_submode=='21':
                split=[2,2,4] #8
            elif mapping_submode=='22':
                split=[2,2,2,2] #8
            elif mapping_submode=='23':
                split=[2,2] #4
            elif mapping_submode=='24':
                split=[1,1,2] #4
            elif mapping_submode=='25':
                split=[1,1,1,1] #4
            elif mapping_submode=='26':
                split=[1,1] #2
            with tf.device('/device:GPU:0'):
                correlator=Correlator(args.latent_dim, mode=mapping_mode, split=split)

    #VAE with structures from DCGAN
    with tf.device('/device:GPU:0'):
        model=CVAE(latent_dim=latent_dim, encoder_params=encoder_params, decoder_params=decoder_params, image_shape=image_shape)


    #Experiment build
    if mapping =='':
        experiment=Experiment.Experiment(model, args.batch_size, None, experiment_id=experiment_id)
    elif mapping=='sparse':
        experiment=Experiment.Sparse_Experiment(model, args.batch_size, dim_selector, experiment_id=experiment_id)
    elif mapping=='correlated':
        experiment=Experiment.Experiment(model, args.batch_size, correlator, experiment_id=experiment_id)
        
    experiment.restore()
    
    #Inference
    batch_size=100
    z_mapped_list=[]
    z_list=[]
    mean_list=[]
    logvar_list=[]
    label_list=[]
    if mapping=='sparse':
        dim_rate_list=[]
    for img, label in test_dataset_with_labels: #!dataset still not out
        label_list.append(label)
        mean, logvar=experiment.model.encode(img)
        z=reparameterize(mean, logvar)
        if mapping!='':
            z_mapped, info=experiment.mapping(z)
        else:
            z_mapped=z
    
        z_mapped_list.append(z_mapped)
        z_list.append(z)
        mean_list.append(mean)
        logvar_list.append(logvar)
        if mapping=='sparse':
            dim_rate_list.append(info[1])
    z_mapped=np.concatenate(z_mapped_list,axis=0)
    z=np.concatenate(z_list,axis=0)
    mean=np.concatenate(mean_list,axis=0) 
    logvar=np.concatenate(logvar_list,axis=0)
    label=np.concatenate(label_list,axis=0)
    return z, z_mapped, label

z_list=[]
z_mapped_list=[]
#![0, 8, 1, 2, 3, -1, -2, -3, -4, -5, -6, -7]
#for i in [0, 8, 1, 2, 3, -1, -2, -3, -4, -5, -6, -7]:
for i in range(len(experiments)):
    z, z_mapped, label = get_feature([experiments[i][0], experiments[i][1], experiments[i][2][0]])
    z_list.append(z)
    z_mapped_list.append(z_mapped)
    
    
##Classification Experiment
log_dir='log/'
make_recursive_dir(log_dir)
experiment_id='celebA_downstream_task'
log_path=log_dir+experiment_id+'_32-2.txt'
log=Log(log_path, append=True)
        
##Classification Experiment
data_size_list=[50, 100, 200, 500, 1000, 2000, 5000, 10000]

for j in range(args.feature_start_id, args.feature_end_id):
    label_input=label[:,j]
    for i in range(len(z_mapped_list)):
        z_input=z_mapped_list[i]
        if j>=40:
            result=classifier_experiment(1, z_input[:-2000],label_input[:-2000], z_input[-2000:],label_input[-2000:], True, data_size_list)
        else:
            result=classifier_experiment(2, z_input[:-2000],label_input[:-2000], z_input[-2000:],label_input[-2000:], False, data_size_list)
        String=str(j)+'\t'+str(i)+'\t'+'\t'.join([str(x) for x in result])
        log(String)
        