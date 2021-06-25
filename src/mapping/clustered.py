import tensorflow as tf
import numpy as np

def cluster_transform(sample, cluster_dim=1):    
    out_list=[]
    for dim in range(cluster_dim):
        inp=sample[:,dim:dim+1]
        out_list.append(tf.cast(tf.math.sign(inp), tf.float32)*tf.abs(inp)**0.2+inp) #!
    out_list.append(sample[:,cluster_dim:])
    return tf.concat(out_list, axis=1)


class Cluster(tf.keras.Model):
    def __init__(self, cluster_dim):
        super(Cluster, self).__init__()
        self.cluster_dim=cluster_dim
            
    def call(self, x):
        x_mapped=cluster_transform(x, self.cluster_dim)
        return x_mapped, []  
    
    def load_weights(self, path):
        pass
    def save_weights(self, path):
        pass
        
if __name__=='__main__':
    cluster=Cluster(2)
    a=np.random.random((20, 10)).astype(np.float32)
    b=cluster(a)[0]
    print(a.shape, b.shape)