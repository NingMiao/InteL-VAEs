import tensorflow as tf
import numpy as np
from time import time
import matplotlib.pyplot as plt
from PIL import Image
import os

from src.mapping.sparse import sparse_metric, sparse_loss_function
from src.misc import make_recursive_dir

#Util Functions
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

def KL_term(mean_1, log_var_1, mean_2, log_var_2):
    dim_KL = 0.5*(log_var_2-log_var_1)+(tf.exp(log_var_1)+(mean_1-mean_2)**2)/(2*tf.exp(log_var_2))-0.5
    return tf.reduce_sum(dim_KL, axis=1)

def reparameterize(mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean
  
def reparameterize_multi(mean, logvar, num=10):
    shape=mean.shape
    shape_new=[mean.shape[0], num]
    for i in range(1, len(shape)):
        shape_new.append(shape[i])
    eps = tf.random.normal(shape=shape_new)   
    return eps * tf.exp(tf.expand_dims(logvar, axis=1) * .5) + tf.expand_dims(mean, axis=1)

def laplace(x, b=0.1):
    return 1/(2*b)*tf.exp(-np.abs(x)/b)

class Experiment:
    def __init__(self, model, batch_size, mapping=None, experiment_id='experiment'):
        self.model=model
        self.mapping=mapping
        self.batch_size=batch_size
        self.experiment_id=experiment_id
        #For generation
        self.random_vector_for_generation = tf.random.normal(shape=[16, self.model.latent_dim])
        
        #For training
        with tf.device('/device:GPU:0'):
            self.optimizer = tf.keras.optimizers.Adam()
        
        #Record
        self.reconstruction_logprob_record = tf.keras.metrics.Mean(name='reconstruction')
        self.KL_term_record = tf.keras.metrics.Mean(name='KL')        
        self.ELBO_record = tf.keras.metrics.Mean(name='ELBO')
        self.loss_record = tf.keras.metrics.Mean(name='loss')
        self.logP_record = tf.keras.metrics.Mean(name='logP')
        
    @tf.function
    def compute_loss(self, x, beta, gamma, loss_type='se', training=False, IWAE=0):
        mean, logvar = self.model.encode(x, training=training)
        if IWAE==0:
            z = reparameterize(mean, logvar)
        else:
            z_multi = reparameterize_multi(mean, logvar, IWAE)
            z = tf.reshape(z_multi, [-1, self.model.latent_dim])
            x = tf.repeat(x, repeats=[IWAE]*x.shape[0], axis=0)
        if self.mapping:
            z, info=self.mapping(z)
        x_logit = self.model.decode(z, training=training)
  
        if loss_type=='cross_entropy':
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
            logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        elif loss_type=='se':
            logpx_z=-tf.reduce_sum((x_logit-x)**2, axis=[1,2,3])
        elif loss_type=='laplace':
            logpx_z=-tf.reduce_sum(laplace(x_logit-x), axis=[1,2,3])
  

        reconstruction = logpx_z
        if IWAE==0:
            KL = KL_term(mean, logvar, mean*0, logvar*0)
            elbo = reconstruction - KL*beta
        else:
            logpz=log_normal_pdf(z, z*0, z*0+1, raxis=-1)
            mean_multi=tf.repeat(mean, repeats=[IWAE]*mean.shape[0], axis=0)
            logvar_multi=tf.repeat(logvar, repeats=[IWAE]*mean.shape[0], axis=0)
            logqz_x=log_normal_pdf(z, mean_multi, logvar_multi, raxis=-1)
            elbo=reconstruction+logpz-logqz_x
        
        if IWAE>0:
            elbo=tf.reshape(elbo,[-1, IWAE])
            elbo_for_exp=elbo-tf.reduce_max(elbo, axis=-1, keepdims=True)
            elbo_exp=tf.exp(elbo_for_exp)
            w=elbo_exp/tf.reduce_sum(elbo_exp, axis=1, keepdims=True)
            w=tf.stop_gradient(w)
            elbo=tf.reduce_sum(w*elbo, axis=1)
            KL=logqz_x-logpz
        
        loss = -tf.reduce_mean(elbo)
        loss_info = []
        return loss, elbo, KL, reconstruction, loss_info

    @tf.function
    def compute_apply_gradients(self, x, beta, gamma, loss_type='se', IWAE=0):
        with tf.GradientTape() as tape:
            loss, elbo, KL, reconstruction, loss_info = self.compute_loss(x, beta, gamma, loss_type='se', training=True, IWAE=IWAE)
        
        trainable_variables=self.model.trainable_variables
        if self.mapping:
            trainable_variables+=self.mapping.trainable_variables
        
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
  
        self.loss_record(loss)
        self.ELBO_record(elbo)
        self.KL_term_record(KL)
        self.reconstruction_logprob_record(reconstruction)
    
    @tf.function
    def compute_log_P(self, x, num=10):
        x_shape = x.shape
        mean, logvar = self.model.encode(x, training=False)
        z = reparameterize_multi(mean, logvar, num)
        
        
        
        mean_multi=tf.tile(tf.expand_dims(mean, axis=1), multiples=[1,num,1])
        log_var_multi=tf.tile(tf.expand_dims(logvar, axis=1), multiples=[1,num,1])
        z_logprob_in_log_z_x=log_normal_pdf(z, mean_multi, log_var_multi, raxis=-1)
        z_logprob_in_log_z=log_normal_pdf(z, mean_multi*0, log_var_multi*0, raxis=-1)
        logratio=z_logprob_in_log_z-z_logprob_in_log_z_x

        z_reshape = tf.reshape(z, [-1, z.shape[-1]])
        
        if self.mapping:
            z_reshape, _ = self.mapping(z_reshape)
            
        x_logit  = self.model.decode(z_reshape)
        x_logit = tf.reshape(x_logit, [x_shape[0], -1, x_shape[1], x_shape[2], x_shape[3]])
        x_multi = tf.tile(tf.expand_dims(x, axis=1), multiples=[1,num,1,1,1])
        logP_x_z=-tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x_multi)
        logP_x_z=tf.reduce_sum(logP_x_z, axis=[-1,-2,-3])
        logP_pre=logP_x_z+logratio
        logP_pre_max=tf.reduce_max(logP_pre, axis=1)
        logP_pre_sub=logP_pre-tf.expand_dims(logP_pre_max, axis=1)
        logP_x=tf.math.log(tf.reduce_mean(tf.exp(logP_pre_sub), axis=1))+logP_pre_max
        self.logP_record(logP_x)
  
    ##Quality check:
    def generate_and_save_images(self, epoch):
        folder='generate_images/'+self.experiment_id+'/'
        make_recursive_dir(folder)
        
        z = self.random_vector_for_generation
        if self.mapping:
            z, _ =self.mapping(z)
        predictions = self.model.decode(z)
        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            if self.model.image_shape[-1]==1:
                plt.imshow(predictions[i, :, :, 0], cmap='gray')
            else:
                img=predictions[i, :, :]
                plt.imshow(img)
            plt.axis('off')
        plt.savefig(folder+'/generate_image_at_epoch_{:04d}.png'.format(epoch))

    def reconstruct_and_save_images(self, epoch, test_dataset):
        folder='reconstruct_images/'+self.experiment_id+'/'
        make_recursive_dir(folder)
        
        x=[]
        i=0
        for batch in test_dataset:
            x.append(batch)
            i+=batch.shape[0]
            if i>=16:
                break
        x=np.concatenate(x, axis=0)[:16]
  
        for i in range(x.shape[0]):
            plt.subplot(4, 4, i+1)
            if self.model.image_shape[-1]==1:
                plt.imshow(x[i, :, :, 0], cmap='gray')
            else:
                img=x[i, :, :]
                plt.imshow(img)
            plt.axis('off')
            plt.savefig(folder+'/image_real.png'.format(epoch))

        mean, logvar = self.model.encode(x)
        if self.mapping:
            mean, _=self.mapping(mean)
        
        predictions = self.model.decode(mean, apply_sigmoid=False)
  
        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            if self.model.image_shape[-1]==1:
                plt.imshow(predictions[i, :, :, 0], cmap='gray')
            else:
                img=predictions[i, :, :]
                plt.imshow(img)
            plt.axis('off')
        plt.savefig(folder+'/reconstruct_image_at_epoch_{:04d}.png'.format(epoch))
        
    def generate_and_save_images_for_FID(self, sample_size, folder=''):
        if folder=='':
            folder='FID_images/'+self.experiment_id+'/'
        make_recursive_dir(folder)
        try:
            os.system('rm '+folder+'*')
        except:
            pass
        sample_list=[]
        for i in range(sample_size//self.batch_size+1):
            test_input= np.random.normal(0,1,[self.batch_size, self.model.latent_dim]).astype(np.float32)
            if self.mapping:
                test_input, info=self.mapping(test_input)
            predictions = self.model.decode(test_input)
            sample_list.append(predictions)
        samples=np.concatenate(sample_list, axis=0)
        ##Some issue about clipping
        if self.model.image_shape[-1]==1:
            samples_max=np.max(samples, axis=(1,2,3), keepdims=True)
            samples_min=np.min(samples, axis=(1,2,3), keepdims=True)
            samples=(samples-samples_min)/(samples_max-samples_min)
            samples_color=np.tile(samples, [1,1,1,3])
        else:
            samples_color=samples
        for i in range(samples_color.shape[0]):
            img=Image.fromarray((samples_color[i]*255).astype(np.uint8))
            img.save(folder+str(i)+'.png')
        return folder
    
    
    #Run Experiment on datasets
    def train_epoch(self, train_dataset, beta, gamma, loss_type='se', IWAE=0):
        #Reset states
        start_time = time()
        self.loss_record.reset_states()
        self.ELBO_record.reset_states()
        self.KL_term_record.reset_states()
        self.reconstruction_logprob_record.reset_states()
        for train_x in train_dataset:
            self.compute_apply_gradients(train_x, beta, gamma, loss_type='se', IWAE=IWAE)
        end_time = time()
        return self.loss_record.result().numpy(), self.ELBO_record.result().numpy(), self.KL_term_record.result().numpy(), self.reconstruction_logprob_record.result().numpy(), end_time-start_time, []
        
  
    def test_epoch(self, test_dataset, beta, gamma, loss_type='se', IWAE=0):
        #Reset states
        start_time = time()
        self.loss_record.reset_states()
        self.ELBO_record.reset_states()
        self.KL_term_record.reset_states()
        self.reconstruction_logprob_record.reset_states()
        for test_x in test_dataset:
            loss, elbo, KL, reconstruction , _ = self.compute_loss(test_x, beta, gamma, loss_type='se', IWAE=IWAE)
            self.loss_record(loss)
            self.ELBO_record(elbo)
            self.KL_term_record(KL)
            self.reconstruction_logprob_record(reconstruction)
        end_time = time()
        return self.loss_record.result().numpy(), self.ELBO_record.result().numpy(), self.KL_term_record.result().numpy(), self.reconstruction_logprob_record.result().numpy(), end_time-start_time, []
    
    #Run Experiment on datasets 
    def sparse_score_epoch(self, test_dataset, test_size=2000):
        test_images=[]
        i=0
        for batch in test_dataset:
            test_images.append(batch)
            i+=batch.shape[0]
            if i>=test_size:
                break
        test_images=np.concatenate(test_images, axis=0)[:test_size]
        mean, logvar = self.model.encode(test_images)
        z = reparameterize(mean, logvar)
        if self.mapping:
            z_mapped, info=self.mapping(z)
        else:
            z_mapped=z
        return sparse_metric(z_mapped)  
    
    def logP_epoch(self, test_dataset_for_logP, num, sample_size):
        self.logP_record.reset_states()
        i=0
        for test_x in test_dataset_for_logP:
            self.compute_log_P(test_x, num)
            i+=test_x.shape[0]
            if i>=sample_size:
                break
        return self.logP_record.result()
 
    
    #Save and Load experiments
    def save(self, epoch=None):
        model_name=self.experiment_id
        if epoch:
            model_name=model_name+'_'+str(epoch)
        
        self.model.save_weights('model/'+model_name+'/CVAE/model.ckpt')
        if self.mapping:
            self.mapping.save_weights('model/'+model_name+'/mapping/model.ckpt')
        
    def restore(self, epoch=None):
        model_name=self.experiment_id
        if epoch:
            model_name=model_name+'_'+str(epoch)
            
        self.model.load_weights('model/'+model_name+'/CVAE/model.ckpt')
        if self.mapping:
            self.mapping.load_weights('model/'+model_name+'/mapping/model.ckpt')
    
class Sparse_Experiment(Experiment):
    #For training with sparse loss
    def __init__(self, model, batch_size, mapping, experiment_id='experiment'):
        super(Sparse_Experiment, self).__init__(model, batch_size, mapping, experiment_id)
        self.sparse_loss_record = tf.keras.metrics.Mean(name='sparse_loss')
    
    @tf.function
    def compute_loss(self, x, beta, gamma, loss_type='se', training=False, IWAE=1):
        mean, logvar = self.model.encode(x, training=training)
        if IWAE==0:
            z = reparameterize(mean, logvar)
        else:
            z_multi = reparameterize_multi(mean, logvar, IWAE)
            z = tf.reshape(z_multi, [-1, self.model.latent_dim])
            x = tf.repeat(x, repeats=[IWAE]*x.shape[0], axis=0)
            
        if self.mapping:
            z, info=self.mapping(z)
            #!if len(info)>0:
            #!    sparse_loss=info[0]
            #!else:
            #!    sparse_loss=sparse_loss_function(tf.abs(z))
            sparse_loss=sparse_loss_function(tf.abs(z))#! This is for generate features.
        else:
            #Notice that the sparse loss is different for vanilla VAE.
            sparse_loss=sparse_loss_function(tf.abs(z))
        x_logit = self.model.decode(z, training=training)
  
        if loss_type=='cross_entropy':
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
            logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        elif loss_type=='se':
            logpx_z=-tf.reduce_sum((x_logit-x)**2, axis=[1,2,3])
        elif loss_type=='laplace':
            logpx_z=-tf.reduce_sum(laplace(x_logit-x), axis=[1,2,3])

        reconstruction = logpx_z
        if IWAE==0:
            KL = KL_term(mean, logvar, mean*0, logvar*0)
            elbo = reconstruction - KL*beta
        else:
            logpz=log_normal_pdf(z, z*0, z*0+1, raxis=-1)
            mean_multi=tf.repeat(mean, repeats=[IWAE]*mean.shape[0], axis=0)
            logvar_multi=tf.repeat(logvar, repeats=[IWAE]*mean.shape[0], axis=0)
            logqz_x=log_normal_pdf(z, mean_multi, logvar_multi, raxis=-1)
            elbo=reconstruction+logpz-logqz_x
        
        if IWAE>0:
            elbo=tf.reshape(elbo,[-1, IWAE])
            elbo_for_exp=elbo-tf.reduce_max(elbo, axis=-1, keepdims=True)
            elbo_exp=tf.exp(elbo_for_exp)
            w=elbo_exp/tf.reduce_sum(elbo_exp, axis=1, keepdims=True)
            w=tf.stop_gradient(w)
            elbo=tf.reduce_sum(w*elbo, axis=1)
            KL=logqz_x-logpz
        
        loss = -tf.reduce_mean(elbo) + sparse_loss*gamma        
        return loss, elbo, KL, reconstruction, [sparse_loss]
        
    @tf.function
    def compute_apply_gradients(self, x, beta, gamma, loss_type='se', IWAE=0):
        with tf.GradientTape() as tape:
            loss, elbo, KL, reconstruction, loss_info = self.compute_loss(x, beta, gamma, loss_type, training=True, IWAE=IWAE) 
        trainable_variables = self.model.trainable_variables
        if self.mapping:
            trainable_variables += self.mapping.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
  
        self.loss_record(loss)
        self.ELBO_record(elbo)
        self.sparse_loss_record(loss_info[0])
        self.KL_term_record(KL)
        self.reconstruction_logprob_record(reconstruction)  
    
    def train_epoch(self, train_dataset, beta, gamma, loss_type='se', IWAE=0):
        #Reset states
        start_time = time()
        self.loss_record.reset_states()
        self.ELBO_record.reset_states()
        self.KL_term_record.reset_states()
        self.reconstruction_logprob_record.reset_states()
        self.sparse_loss_record.reset_states()
        for train_x in train_dataset:
           self.compute_apply_gradients(train_x, beta, gamma, loss_type='se', IWAE=IWAE) 
        end_time = time()
        return self.loss_record.result().numpy(), self.ELBO_record.result().numpy(), self.KL_term_record.result().numpy(), self.reconstruction_logprob_record.result().numpy(), end_time-start_time, [self.sparse_loss_record.result().numpy()]
        
  
    def test_epoch(self, test_dataset, beta, gamma, loss_type='se', IWAE=0):
        #Reset states
        start_time = time()
        self.loss_record.reset_states()
        self.ELBO_record.reset_states()
        self.KL_term_record.reset_states()
        self.reconstruction_logprob_record.reset_states()
        self.sparse_loss_record.reset_states()
        for test_x in test_dataset:
            loss, elbo, KL, reconstruction , info = self.compute_loss(test_x, beta, gamma, loss_type='se', IWAE=IWAE)
            self.loss_record(loss)
            self.ELBO_record(elbo)
            self.KL_term_record(KL)
            self.reconstruction_logprob_record(reconstruction)
            self.sparse_loss_record(info[0])
        end_time = time()
        return self.loss_record.result().numpy(), self.ELBO_record.result().numpy(), self.KL_term_record.result().numpy(), self.reconstruction_logprob_record.result().numpy(), end_time-start_time, [self.sparse_loss_record.result().numpy()]

class Vamp_Experiment(Sparse_Experiment):
    #For training with vamp prior
    def __init__(self, model, batch_size, mapping, experiment_id='experiment', vamp_input_num=10):
        super(Vamp_Experiment, self).__init__(model, batch_size, mapping, experiment_id)
        self.sparse_loss_record = tf.keras.metrics.Mean(name='sparse_loss')
        initial_value=np.random.normal(0,1,[vamp_input_num]+ list(model.image_shape)).astype(np.float32)
        self.pseduo_input=tf.Variable(initial_value=initial_value, dtype=tf.float32)
        #with tf.device('/device:GPU:0'): 
            #Selects different learning rate for vamp. Not working.
        #    self.optimizer = tf.keras.optimizers.Adam(3e-3)
        
    
    def logP_Gaussian(self, x, mean, log_var):
        #utils
        dim = tf.shape(x)[1]
        return -tf.cast(dim, tf.float32)*0.5*(tf.math.log(2.0*np.pi)+tf.reduce_sum(log_var, axis=1))-0.5*tf.reduce_sum((x-mean)**2*tf.exp(-log_var), axis=1)
    def logP_MoG(self, x, mean, log_var):
        #utils
        x=tf.expand_dims(x, 1)
        mean=tf.expand_dims(mean, 0)
        log_var=tf.expand_dims(log_var, 0)
        logP_component_dim=-np.log(2*np.pi)/2-log_var/2-(x-mean)**2/(2*tf.exp(log_var))
        logP_component=tf.reduce_sum(logP_component_dim, axis=-1)
        def logsumexp(x, axis=-1):
            x_max=tf.reduce_max(x, axis=axis, keepdims=True)
            x-=x_max
            x_exp=tf.exp(x)
            x_exp_mean=tf.reduce_sum(x_exp, axis=axis)
            x_exp_mean_log=tf.math.log(x_exp_mean)+tf.squeeze(x_max, axis=-1)
            return x_exp_mean_log
        def logmeanexp(x, axis=-1):
            return logsumexp(x, axis=axis)-tf.math.log(float(x.shape[axis]))
        logP=logmeanexp(logP_component, axis=-1)
        return logP
    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        #utils
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)
    def generate_from_MoG(self, batch_size, mean, log_var):
        #utils
        samples=[]
        batch_size_each_component=int(np.ceil(batch_size/mean.shape[0]))
        for i in range(mean.shape[0]):
            sample=np.random.normal(0, 1, [batch_size_each_component, mean.shape[1]]).astype(np.float32)
            sample=sample*np.exp(log_var[i]*0.5)+mean[i]
            samples.append(sample)
        samples=np.concatenate(samples, axis=0)
        np.random.shuffle(samples)
        return samples[:batch_size]

    @tf.function
    def compute_loss(self, x, beta, gamma, loss_type='se', training=False):
        mean, logvar = self.model.encode(x, training=training)
        z = reparameterize(mean, logvar)
        if self.mapping:
            z, info=self.mapping(z)
            sparse_loss=info[0]
        else:
            #Notice that the sparse loss is different for vanilla VAE.
            sparse_loss=sparse_loss_function(tf.abs(z))
        x_logit = self.model.decode(z, training=training)
  
        if loss_type=='cross_entropy':
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
            logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        elif loss_type=='se':
            logpx_z=-tf.reduce_sum((x_logit-x)**2, axis=[1,2,3])
        elif loss_type=='laplace':
            logpx_z=-tf.reduce_sum(laplace(x_logit-x), axis=[1,2,3])

        reconstruction = logpx_z
        
        if True:
            pseduo_z_x_mean, pseduo_z_x_log_var = self.model.encode(self.pseduo_input, training=training)
            
            logQ = self.logP_Gaussian(z, mean, logvar)
            logP = self.logP_MoG(z, pseduo_z_x_mean, pseduo_z_x_log_var)
            KL = tf.reduce_mean(logQ-logP)
        
        elbo = reconstruction - KL*beta
        loss = -tf.reduce_mean(elbo) + sparse_loss*gamma
        return loss, elbo, KL, reconstruction, [sparse_loss]
    
    @tf.function
    def compute_apply_gradients(self, x, beta, gamma, loss_type='se'):
        with tf.GradientTape() as tape:
            loss, elbo, KL, reconstruction, loss_info = self.compute_loss(x, beta, gamma, loss_type, training=True)
        trainable_variables = self.model.trainable_variables
        trainable_variables += [self.pseduo_input] #For vamp
        if self.mapping:
            trainable_variables += self.mapping.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
  
        self.loss_record(loss)
        self.ELBO_record(elbo)
        self.sparse_loss_record(loss_info[0])
        self.KL_term_record(KL)
        self.reconstruction_logprob_record(reconstruction)  
    
    @tf.function
    def compute_log_P(self, x, num=10):
        x_shape = x.shape
        mean, logvar = self.model.encode(x, training=False)
        z = reparameterize_multi(mean, logvar, num)        
        
        mean_multi=tf.tile(tf.expand_dims(mean, axis=1), multiples=[1,num,1])
        log_var_multi=tf.tile(tf.expand_dims(logvar, axis=1), multiples=[1,num,1])
        z_logprob_in_log_z_x=self.log_normal_pdf(z, mean_multi, log_var_multi, raxis=-1)
        #z_logprob_in_log_z=log_normal_pdf(z, mean_multi*0, log_var_multi*0, raxis=-1)
        
        #For vamp
        pseduo_z_x_mean, pseduo_z_x_log_var = self.model.encode(self.pseduo_input, training=False)
        z_reshape=tf.reshape(z, [-1,self.model.latent_dim])
        z_logprob_in_log_z=self.logP_MoG(z_reshape, pseduo_z_x_mean, pseduo_z_x_log_var)
        z_logprob_in_log_z=tf.reshape(z_logprob_in_log_z, [-1, num])
        
        
        logratio=z_logprob_in_log_z-z_logprob_in_log_z_x

        z_reshape = tf.reshape(z, [-1, z.shape[-1]])
        
        if self.mapping:
            z_reshape, _ = self.mapping(z_reshape)
            
        x_logit  = self.model.decode(z_reshape)
        x_logit = tf.reshape(x_logit, [x_shape[0], -1, x_shape[1], x_shape[2], x_shape[3]])
        x_multi = tf.tile(tf.expand_dims(x, axis=1), multiples=[1,num,1,1,1])
        logP_x_z=-tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x_multi)
        logP_x_z=tf.reduce_sum(logP_x_z, axis=[-1,-2,-3])
        logP_pre=logP_x_z+logratio
        logP_pre_max=tf.reduce_max(logP_pre, axis=1)
        logP_pre_sub=logP_pre-tf.expand_dims(logP_pre_max, axis=1)
        logP_x=tf.math.log(tf.reduce_mean(tf.exp(logP_pre_sub), axis=1))+logP_pre_max
        self.logP_record(logP_x)
    
    def generate_and_save_images(self, epoch):
        folder='generate_images/'+self.experiment_id+'/'
        make_recursive_dir(folder)
        
        pseduo_z_x_mean, pseduo_z_x_log_var = self.model.encode(self.pseduo_input)
        z=self.generate_from_MoG(16, pseduo_z_x_mean, pseduo_z_x_log_var)
        if self.mapping:
            z, _ =self.mapping(z)
        predictions = self.model.decode(z)
        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            if self.model.image_shape[-1]==1:
                plt.imshow(predictions[i, :, :, 0], cmap='gray')
            else:
                img=predictions[i, :, :]
                plt.imshow(img)
            plt.axis('off')
        plt.savefig(folder+'/generate_image_at_epoch_{:04d}.png'.format(epoch))

class MoG_Experiment(Sparse_Experiment):
    #For training with vamp prior
    def __init__(self, model, batch_size, mapping, experiment_id='experiment', MoG_num=2):
        ##Only support the case where MoG_num=2 or 4.
        super(MoG_Experiment, self).__init__(model, batch_size, mapping, experiment_id)
        self.sparse_loss_record = tf.keras.metrics.Mean(name='sparse_loss')
        
        initial_value=np.zeros([MoG_num, model.latent_dim]).astype(np.float32)
        if MoG_num==2:
            initial_value[0,0]=1
            initial_value[1,0]=-1
        elif MoG_num==4:
            initial_value[0,0]=1
            initial_value[1,0]=-1
            initial_value[2,1]=1
            initial_value[3,1]=-1
        initial_value*=2
        self.MoG_mean=initial_value
        self.MoG_logvar=initial_value*0+1.0
        
    
    def logP_Gaussian(self, x, mean, log_var):
        #utils
        dim = tf.shape(x)[1]
        return -tf.cast(dim, tf.float32)*0.5*(tf.math.log(2.0*np.pi)+tf.reduce_sum(log_var, axis=1))-0.5*tf.reduce_sum((x-mean)**2*tf.exp(-log_var), axis=1)
    def logP_MoG(self, x, mean, log_var):
        #utils
        x=tf.expand_dims(x, 1)
        mean=tf.expand_dims(mean, 0)
        log_var=tf.expand_dims(log_var, 0)
        logP_component_dim=-np.log(2*np.pi)/2-log_var/2-(x-mean)**2/(2*tf.exp(log_var))
        logP_component=tf.reduce_sum(logP_component_dim, axis=-1)
        def logsumexp(x, axis=-1):
            x_max=tf.reduce_max(x, axis=axis, keepdims=True)
            x-=x_max
            x_exp=tf.exp(x)
            x_exp_mean=tf.reduce_sum(x_exp, axis=axis)
            x_exp_mean_log=tf.math.log(x_exp_mean)+tf.squeeze(x_max, axis=-1)
            return x_exp_mean_log
        def logmeanexp(x, axis=-1):
            return logsumexp(x, axis=axis)-tf.math.log(float(x.shape[axis]))
        logP=logmeanexp(logP_component, axis=-1)
        return logP
    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        #utils
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)
    def generate_from_MoG(self, batch_size, mean, log_var):
        #utils
        samples=[]
        batch_size_each_component=int(np.ceil(batch_size/mean.shape[0]))
        for i in range(mean.shape[0]):
            sample=np.random.normal(0, 1, [batch_size_each_component, mean.shape[1]]).astype(np.float32)
            sample=sample*np.exp(log_var[i]*0.5)+mean[i]
            samples.append(sample)
        samples=np.concatenate(samples, axis=0)
        np.random.shuffle(samples)
        return samples[:batch_size]

    @tf.function
    def compute_loss(self, x, beta, gamma, loss_type='se', training=False):
        mean, logvar = self.model.encode(x, training=training)
        z = reparameterize(mean, logvar)
        if self.mapping:
            z, info=self.mapping(z)
            sparse_loss=info[0]
        else:
            #Notice that the sparse loss is different for vanilla VAE.
            sparse_loss=sparse_loss_function(tf.abs(z))
        x_logit = self.model.decode(z, training=training)
  
        if loss_type=='cross_entropy':
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
            logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        elif loss_type=='se':
            logpx_z=-tf.reduce_sum((x_logit-x)**2, axis=[1,2,3])
        elif loss_type=='laplace':
            logpx_z=-tf.reduce_sum(laplace(x_logit-x), axis=[1,2,3])

        reconstruction = logpx_z
        
        if True:
            
            logQ = self.logP_Gaussian(z, mean, logvar)
            logP = self.logP_MoG(z, self.MoG_mean, self.MoG_logvar)
            KL = tf.reduce_mean(logQ-logP)
        
        elbo = reconstruction - KL*beta
        loss = -tf.reduce_mean(elbo) + sparse_loss*gamma
        return loss, elbo, KL, reconstruction, [sparse_loss]
    
    
    @tf.function
    def compute_log_P(self, x, num=10):
        x_shape = x.shape
        mean, logvar = self.model.encode(x, training=False)
        z = reparameterize_multi(mean, logvar, num)        
        
        mean_multi=tf.tile(tf.expand_dims(mean, axis=1), multiples=[1,num,1])
        log_var_multi=tf.tile(tf.expand_dims(logvar, axis=1), multiples=[1,num,1])
        z_logprob_in_log_z_x=self.log_normal_pdf(z, mean_multi, log_var_multi, raxis=-1)
        #z_logprob_in_log_z=log_normal_pdf(z, mean_multi*0, log_var_multi*0, raxis=-1)
        
        #For vamp
        z_reshape=tf.reshape(z, [-1,self.model.latent_dim])
        z_logprob_in_log_z=self.logP_MoG(z_reshape, self.MoG_mean, self.MoG_logvar)
        z_logprob_in_log_z=tf.reshape(z_logprob_in_log_z, [-1, num])
        
        
        logratio=z_logprob_in_log_z-z_logprob_in_log_z_x

        z_reshape = tf.reshape(z, [-1, z.shape[-1]])
        
        if self.mapping:
            z_reshape, _ = self.mapping(z_reshape)
            
        x_logit  = self.model.decode(z_reshape)
        x_logit = tf.reshape(x_logit, [x_shape[0], -1, x_shape[1], x_shape[2], x_shape[3]])
        x_multi = tf.tile(tf.expand_dims(x, axis=1), multiples=[1,num,1,1,1])
        logP_x_z=-tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x_multi)
        logP_x_z=tf.reduce_sum(logP_x_z, axis=[-1,-2,-3])
        logP_pre=logP_x_z+logratio
        logP_pre_max=tf.reduce_max(logP_pre, axis=1)
        logP_pre_sub=logP_pre-tf.expand_dims(logP_pre_max, axis=1)
        logP_x=tf.math.log(tf.reduce_mean(tf.exp(logP_pre_sub), axis=1))+logP_pre_max
        self.logP_record(logP_x)
    
    def generate_and_save_images(self, epoch):
        folder='generate_images/'+self.experiment_id+'/'
        make_recursive_dir(folder)
        
        z=self.generate_from_MoG(16, self.MoG_mean, self.MoG_logvar)
        if self.mapping:
            z, _ =self.mapping(z)
        predictions = self.model.decode(z)
        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            if self.model.image_shape[-1]==1:
                plt.imshow(predictions[i, :, :, 0], cmap='gray')
            else:
                img=predictions[i, :, :]
                plt.imshow(img)
            plt.axis('off')
        plt.savefig(folder+'/generate_image_at_epoch_{:04d}.png'.format(epoch))        
        

    
if __name__=='__main__':
    from src.CVAE import CVAE
    model=CVAE()
    #ex=Experiment(model, 20)
    ex=Vamp_Experiment(model, 20, None)