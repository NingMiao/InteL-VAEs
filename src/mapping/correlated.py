import tensorflow as tf
import numpy as np

class Correlator(tf.keras.Model):
    def __init__(self, dim_num, mode=0, split=5, mapping_matrix=None):
        super(Correlator, self).__init__()
        self.mode=mode
        self.dim_num=dim_num
        
        if type(split)==list:
            self.split=split
        else:
            self.split=[]
            dim_each=dim_num//split
            for i in range(split):
                if i!=split-1:
                    self.split.append(dim_each)
                else:
                    self.split.append(dim_num-dim_each*(split-1))
        
        if mode==1:
            self.d1 = tf.keras.layers.Dense(dim_num, activation='relu')
        elif mode in [2,3,4]:
            self.d1=[]
            if mode in [3,4]:
                layer_out_dim_list=self.split[1:]
            else:
                layer_out_dim_list=self.split
            for dim_each in layer_out_dim_list:
                self.d1.append(tf.keras.layers.Dense(dim_each, activation='relu'))
        elif mode in [5]:
            #Fixed mapping
            self.mapping_matrix=mapping_matrix
            
    def call(self, x):
        if self.mode==0:
            out=x
        if self.mode==1:
            out = self.d1(x)
        elif self.mode==2:
            out=[]
            cur_pos=0
            for i in range(len(self.split)):
                out.append(self.d1[i](x[:,cur_pos: cur_pos+self.split[i]]))
                cur_pos+=self.split[i]
            out=tf.concat(out, axis=1)
        elif self.mode==3:
            out=[]
            cur_pos=0
            for i in range(len(self.split)):
                if i==0:
                    out.append(x[:, cur_pos:cur_pos+self.split[i]])
                    
                else:
                    In=tf.concat([out[-1], x[:,cur_pos:cur_pos+self.split[i]]], axis=1)
                    out.append(self.d1[i-1](In))
                cur_pos+=self.split[i]
            out=tf.concat(out, axis=1)
        elif self.mode==4:
            cur_pos=0
            out=x[:, cur_pos:cur_pos+self.split[0]]
            cur_pos+=self.split[0]
            for i in range(1, self.split_num):
                In=tf.concat([out, x[:,cur_pos:cur_pos+self.split[i]]], axis=1)
                out = self.d1[i-1](In)
                cur_pos+=self.split[i]
                
        elif self.mode==5:
            out=tf.matmul(x, self.mapping_matrix)
                
        return out, []  
    
if __name__=='__main__':
    correlator=Correlator(10, 3, 2)
    a=np.random.random((20, 10))
    b=correlator(a)[0]
    print(a.shape, b.shape)