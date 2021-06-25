import tensorflow as tf
import numpy as np

def sparse_loss_entropy(dim_rate):
    dim_rate = dim_rate / tf.reduce_sum(dim_rate, axis=1, keepdims=True)
    return -tf.reduce_mean(tf.reduce_sum(dim_rate*tf.math.log(dim_rate+1e-8), axis=1))
def sparse_loss_poly(dim_rate):
    return tf.reduce_mean(tf.reduce_sum(tf.abs(dim_rate)**0.5, axis=1))
def sparse_loss_relative_entropy(dim_rate):
    dim_rate = dim_rate / tf.reduce_sum(dim_rate, axis=1, keepdims=True)
    mean_each_entropy=-tf.reduce_mean(tf.reduce_sum(dim_rate*tf.math.log(dim_rate+1e-8), axis=1))
    mean_prob=tf.reduce_mean(dim_rate, axis=0)
    mean_prob_entropy=-tf.reduce_sum(mean_prob*tf.math.log(mean_prob+1e-5))
    return - mean_prob_entropy + mean_each_entropy

#Choose between given sparse measurements
sparse_loss_function=sparse_loss_relative_entropy

class Dim_selector(tf.keras.Model):
    def __init__(self, dim_num):
        super(Dim_selector, self).__init__()
        self.mapping_type='sparse'
        self.d1 = tf.keras.layers.Dense(10, activation='relu')
        self.d2 = tf.keras.layers.Dense(10, activation='relu')
        self.d3 = tf.keras.layers.Dense(dim_num, activation=None)
    def __call__(self, x):
        inp = x
        inp = self.d1(inp)
        inp = self.d2(inp)
        inp = self.d3(inp)
        dim_rate = tf.math.sigmoid(inp)*0.8+0.2 #!#!#! change to soft sparse
        ##Sparse loss is flexible
        sparse_loss = sparse_loss_function(dim_rate)
        x_mapped=x*dim_rate
        info=[sparse_loss, dim_rate]
        return x_mapped, info

#Metric in Emile's paper
def sparse_metric(z):
    d=z.shape[1]
    z/=np.std(z, axis=0, keepdims=True)
    r=np.linalg.norm(z, 1, axis=1)/np.linalg.norm(z, 2, axis=1)
    r_mean=np.mean(r)
    return (d**0.5-r_mean)/(d**0.5-1)    
    
if __name__=='__main__':
    dim_selector=Dim_selector(10)
    a=np.random.random((20, 10))
    b, info=dim_selector(a)
    print(b-a*info[1])