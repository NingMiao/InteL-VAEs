import os
import numpy as np

class Log():
    def __init__(self, path, append=False):
        self.path=path
        print('Save log at: '+path)
        if append is False:
            with open(self.path, 'w') as g:
                g.write('')
    def __call__(self, string):
        with open(self.path, 'a') as g:
            g.write(string+'\n')

def make_recursive_dir(dir_name):
    dir_name_list=[dir_name]
    flag=0
    i=0
    while True:
        i+=1
        if i>10:
            break
        if os.path.exists(dir_name_list[-1]) or dir_name_list[-1] in ['', '.']:
            del(dir_name_list[-1])
            flag=1
        else:
            if flag==1:
                try:
                    os.mkdir(dir_name_list[-1])
                    del(dir_name_list[-1])
                except:
                    break
            else:
                dir_name_list.append(os.path.dirname(dir_name_list[-1]))
        if dir_name_list==[]:
            break

def extend_matrix(mat, target_dim):
    mat_out=np.eye(target_dim).astype(np.float32)
    mat_out[:mat.shape[0],:mat.shape[1]]=mat
    return mat_out
            
if __name__=='__main__':
    make_recursive_dir('panda/dog/cat')