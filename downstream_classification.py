import tensorflow as tf

import os
import sys
from time import time
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
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
from src.experiment import reparameterize

parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--latent_dim", type=int, default=2) #50 for MNIST/FashionMNIST, 128 for CelebA, 5 for dsprites
parser.add_argument("--epoch", type=int, default=50)   #30 epochs for celebA
parser.add_argument("--least_epoch", type=int, default=20)   #30 epochs for celebA
parser.add_argument("--mapping", type=str, default='') #['', 'sparse','clustered','correlated']
parser.add_argument("--mapping_mode", type=int, default=5) #Only For correlated: [0,1,2,3,4,5]
parser.add_argument("--mapping_submode", type=int, default=5) #Only For correlated: [0,1,2,3,4,5]
parser.add_argument("--gamma", type=float, default=0.0)
parser.add_argument("--gpu", type=str, default='')
parser.add_argument("--dataset", type=str, default='fashion_mnist') #['mnist', 'fashion_mnist', 'cifar10', 'celebA', 'dsprites']
parser.add_argument("--cut_by_labels_dim", type=int, default=0) #[0, 1, 2...]
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
parser.add_argument("--leading_metric", type=str, default='FID')  #FID, logP, loss

#Test mode
parser.add_argument("--test_only", action='store_true', default=False)
parser.add_argument("--test_dataset_seed", type=int, default=43)
args = parser.parse_args()

##classification experiment (Decision Tree)
def classifier_experiment(z_input, y_input):
    result_list=[]
    for data_size in [10,20, 50, 100, 500, 1000, 2000, 5000]:
        eval_accuracy=decision_tree(z_input, y_input, data_size, 2000)
        
        print('Data size: {}, eval accuracy: {}'.format(data_size, eval_accuracy))
        result_list.append(eval_accuracy.numpy())
    return result_list


#Please use the correct info_id
args.info=['0', '1','2','3','4','5','6','7','8','9']
args.latent_dim=50
args.restore_epoch=30

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
if args.dataset=='mnist' and args.cut_by_labels_dim!=0: #Only load sample with certain label
    labels_for_mnist=list(np.arange(2**args.cut_by_labels_dim).astype(np.int32))
else:
    labels_for_mnist=[]

#train_dataset, test_dataset, test_dataset_for_logP, image_shape=load_dataset(args.dataset, args.batch_size, args.logP_batch_size, labels_for_mnist=labels_for_mnist, test_dataset_seed=args.test_dataset_seed)
if args.dataset in ['mnist','fashion_mnist']:
    test_dataset_with_labels, image_shape=load_dataset(args.dataset, args.batch_size, analysis_mode=True, manual_features=False, class_weighting=True)
elif args.dataset == 'celebA':
    test_dataset_with_labels, image_shape, label_names=load_dataset(args.dataset, args.batch_size, analysis_mode=True)

channel=image_shape[-1]


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
        with tf.device('/device:GPU:0'):
            correlator=Correlator(args.latent_dim, mode=args.mapping_mode, split=split)
elif args.mapping=='clustered':
    with tf.device('/device:GPU:0'):
        cluster=Cluster(args.mapping_mode)
        
        
def reorder(x_label):
    ind_by_label=[]
    for i in range(10):
        ind_by_label.append([])
    for i in range(len(x_label)):
        label=int(x_label[i])
        ind_by_label[label].append(i)
    ind_reorder=[]
    while len(ind_reorder)<len(x_label):
        for i in range(len(ind_by_label)):
            if len(ind_by_label[i])>0:
                ind_reorder.append(ind_by_label[i][0])
                del(ind_by_label[i][0])
    return ind_reorder

#!#!log=Log('log/downstream_activation_status_'+args.dataset+'.txt', append=True)#!#!
log=Log('log/downstream_'+args.dataset+'.txt', append=True)


for info in args.info:
    #Id 
    if args.mapping=='correlated':
        experiment_id=str(args.dataset)+'_'+str(args.mapping)+'_'+str(args.latent_dim)+'_'+str(args.beta)+'_'+str(args.gamma)+'_'+str(args.mapping_mode)+'_'+str(args.mapping_submode)+'_'+info
    else:
        experiment_id=str(args.dataset)+'_'+str(args.mapping)+'_'+str(args.latent_dim)+'_'+str(args.beta)+'_'+str(args.gamma)+'_'+info
    


    #VAE with structures from DCGAN
    encoder_params, decoder_params = hparams.encoder_decoder_params(args.dataset)
    with tf.device('/device:GPU:0'):
        model=CVAE(latent_dim=args.latent_dim, encoder_params=encoder_params, decoder_params=decoder_params, image_shape=image_shape)

    #Experiment build
    if args.mapping=='':
        experiment=Experiment.Experiment(model, args.batch_size, None, experiment_id=experiment_id)
    elif args.mapping=='sparse':
        experiment=Experiment.Sparse_Experiment(model, args.batch_size, dim_selector, experiment_id=experiment_id)
    elif args.mapping=='correlated':
        experiment=Experiment.Experiment(model, args.batch_size, correlator, experiment_id=experiment_id)
    elif args.mapping=='clustered':
        experiment=Experiment.Experiment(model, args.batch_size, cluster, experiment_id=experiment_id)

    if args.restore_epoch>0:
        experiment.restore(args.restore_epoch)
    else:
        experiment.restore()
    
    batch_size=100
    z_mapped_list=[]
    z_list=[]
    mean_list=[]
    logvar_list=[]
    label_list=[]
    feature_list=[]
    if args.mapping=='sparse':
        dim_rate_list=[]
    for img, label in test_dataset_with_labels: #!dataset still not out
        label_list.append(label)
        #feature_list.append(feature)
        mean, logvar=experiment.model.encode(img)
        z=mean
        if args.mapping!='':
            z_mapped, infos=experiment.mapping(z)
        else:
            z_mapped=z
    
        z_mapped_list.append(z_mapped)
        z_list.append(z)
        mean_list.append(mean)
        logvar_list.append(logvar)
    if args.mapping=='sparse':
        dim_rate_list.append(infos[1])
    z_mapped=np.concatenate(z_mapped_list,axis=0)
    z=np.concatenate(z_list,axis=0)
    mean=np.concatenate(mean_list,axis=0) 
    logvar=np.concatenate(logvar_list,axis=0)
    label=np.concatenate(label_list,axis=0)
    if args.mapping=='sparse':
        dim_rate=np.concatenate(dim_rate_list,axis=0)

    #Reorder for classification    
    ind_by_label = reorder(label)
    z_mapped=z_mapped[ind_by_label]
    z=z[ind_by_label]
    label=label[ind_by_label]
    
    ##Classification Experiment
    
    z_input=z_mapped[:,:]
    y_input=np.squeeze(label)
    result=classifier_experiment(z_input, y_input)
    if args.mapping=='sparse':
        print(args.beta)
        print(args.gamma)
        print(info)
        print('\t'.join([str(x) for x in result]))
        string='sparse\t'+str(args.beta)+'\t'+str(args.gamma)+'\t'+info+'\t'+'\t'.join([str(x) for x in result])
    else:
        string='vanilla\t'+str(args.beta)+'\t'+info+'\t'+'\t'.join([str(x) for x in result])
    log(string)
