import tensorflow as tf
import os
import numpy as np
from src.manual_features import feature, weighting
from src.manual_features import class_weighting as c_w

def load_dataset(dataset, batch_size, logP_batch_size=5, binary=False, analysis_mode=False, labels_for_mnist=[], test_dataset_seed=0, manual_features=False, class_weighting=False, with_labels=False):
    
    #Please take care, 'labels_for_mnist' only works on mnist.
    #manual_features and class_weighting work for mnist and fashion_mnist, adding some manual features for analysis
    
    if dataset=='mnist':
        validation=5000
        (train_images, train_labels), (test_images, test_labels) =tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
        image_shape=train_images.shape[1:]
        
        if labels_for_mnist:
            train_each=5000
            validation_each=500
            test_each=1000
            train_count=np.zeros([len(labels_for_mnist)])
            train_ind=[]
            validation_ind=[]
            for i in range(train_labels.shape[0]):
                if train_labels[i] in labels_for_mnist:
                    id_=labels_for_mnist.index(train_labels[i])
                    if train_count[id_]<train_each:
                        train_ind.append(i)
                    elif train_count[id_]<train_each+validation_each:
                        validation_ind.append(i)
                    train_count[id_]+=1
                        
            train_images=train_images[train_ind+validation_ind]
            train_labels=train_labels[train_ind+validation_ind]
            
            test_ind=[]
            test_count=np.zeros([len(labels_for_mnist)])
            for i in range(test_labels.shape[0]):
                if test_labels[i] in labels_for_mnist:
                    id_=labels_for_mnist.index(test_labels[i])
                    if test_count[id_]<test_each:
                        test_ind.append(i)
                    test_count[id_]+=1
            test_images=test_images[test_ind]
            test_labels=test_labels[test_ind]
            print(len(train_images), len(test_images))
        
        #re-range to [0, 1]
        train_images /= 255.
        test_images /= 255.
        if binary:
            train_images[train_images >= .5] = 1.
            train_images[train_images < .5] = 0.
            test_images[test_images >= .5] = 1.
            test_images[test_images < .5] = 0.
            
        TRAIN_BUF = train_images.shape[0]
        TEST_BUF =  test_images.shape[0]
        
        #np.random.seed(test_dataset_seed)
        #test_inds=np.arange(TEST_BUF)
        #np.random.shuffle(test_inds)
        #test_images=test_images[test_inds]
        #test_labels=test_labels[test_inds]
        
        
        if analysis_mode:
            test_labels = test_labels.reshape(test_labels.shape[0], 1).astype(np.float32)
            
            if manual_features:
                features=feature(test_images)
                if class_weighting:
                    features=c_w(test_labels, features)
                else:
                    features=weighting(features)
            
                test_dataset_with_labels = tf.data.Dataset.from_tensor_slices((test_images, test_labels, features)).batch(batch_size)
            else:
                test_dataset_with_labels = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)
            return test_dataset_with_labels, image_shape
        
        if not with_labels:
            if validation>0:
                train_dataset = tf.data.Dataset.from_tensor_slices(train_images[:-validation]).shuffle(TRAIN_BUF-validation).batch(batch_size)
                validation_dataset = tf.data.Dataset.from_tensor_slices(train_images[-validation:]).batch(batch_size)
            else:
                train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(batch_size)
            test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)
        else:
            if validation>0:
                train_dataset = tf.data.Dataset.from_tensor_slices((train_images[:-validation], train_labels[:-validation])).shuffle(TRAIN_BUF-validation).batch(batch_size)
                validation_dataset = tf.data.Dataset.from_tensor_slices((train_images[-validation:], train_labels[-validation:])).batch(batch_size)
            else:
                train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(TRAIN_BUF).batch(batch_size)
            test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)
        
        if logP_batch_size>0:
            test_dataset_for_logP=tf.data.Dataset.from_tensor_slices(test_images).batch(logP_batch_size)
        else:
            test_dataset_for_logP=[]
        if validation>0:
            return train_dataset, test_dataset, test_dataset_for_logP, image_shape, validation_dataset
        else:
            return train_dataset, test_dataset, test_dataset_for_logP, image_shape
            
    elif dataset=='fashion_mnist':
        validation=5000
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
        image_shape=train_images.shape[1:]
        
        #re-range to [0, 1]
        train_images /= 255.
        test_images /= 255.
        if binary:
            train_images[train_images >= .5] = 1.
            train_images[train_images < .5] = 0.
            test_images[test_images >= .5] = 1.
            test_images[test_images < .5] = 0.
        
        TRAIN_BUF = train_images.shape[0]
        TEST_BUF =  test_images.shape[0]
        
        #np.random.seed(test_dataset_seed)
        #test_inds=np.arange(TEST_BUF)
        #np.random.shuffle(test_inds)
        #test_images=test_images[test_inds]
        #test_labels=test_labels[test_inds]
        
        if analysis_mode:
            test_labels = test_labels.reshape(test_labels.shape[0], 1).astype(np.float32)
            if manual_features:
                features=feature(test_images)
                if class_weighting:
                    features=c_w(test_labels, features)
                else:
                    features=weighting(features)
            
                test_dataset_with_labels = tf.data.Dataset.from_tensor_slices((test_images, test_labels, features)).batch(batch_size)
            else:
                test_dataset_with_labels = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)
            return test_dataset_with_labels, image_shape
        
        if validation>0:
            train_dataset = tf.data.Dataset.from_tensor_slices(train_images[:-validation]).shuffle(TRAIN_BUF-validation).batch(batch_size)
            validation_dataset = tf.data.Dataset.from_tensor_slices(train_images[-validation:]).batch(batch_size)
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)

        if logP_batch_size>0:
            test_dataset_for_logP=tf.data.Dataset.from_tensor_slices(test_images).batch(logP_batch_size)
        else:
            test_dataset_for_logP=[]
        
        if validation>0:
            return train_dataset, test_dataset, test_dataset_for_logP, image_shape, validation_dataset
        else:
            return train_dataset, test_dataset, test_dataset_for_logP, image_shape
        
    elif dataset=='cifar10':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
        train_images = train_images.reshape(train_images.shape[0], 32,32,3).astype('float32')
        test_images = test_images.reshape(test_images.shape[0], 32,32,3).astype('float32')
        image_shape=train_images.shape[1:]
        
        #re-range to [0, 1]
        train_images /= 255.
        test_images /= 255.
        if binary:
            train_images[train_images >= .5] = 1.
            train_images[train_images < .5] = 0.
            test_images[test_images >= .5] = 1.
            test_images[test_images < .5] = 0.

        TRAIN_BUF = train_images.shape[0]
        TEST_BUF =  test_images.shape[0]
        
        np.random.seed(test_dataset_seed)
        test_inds=np.arange(TEST_BUF)
        np.random.shuffle(test_inds)
        test_images=test_images[test_inds]
        test_labels=test_labels[test_inds]
        
        if analysis_mode:
            test_labels = test_labels.reshape(test_labels.shape[0], 1).astype(np.float32)
            test_dataset_with_labels = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)
            return test_dataset_with_labels, image_shape
        
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)

        if logP_batch_size>0:
            test_dataset_for_logP=tf.data.Dataset.from_tensor_slices(test_images).batch(logP_batch_size)
        else:
            test_dataset_for_logP=[]
        return train_dataset, test_dataset, test_dataset_for_logP, image_shape
    
    elif dataset=='celebA':
        validation=19687
        data_dir='datasets/celebA/'
        
        if not os.path.exists(data_dir):
            #Preprocess functions
            os.mkdir(data_dir)
            key_list=['image', 'attributes', 'landmarks']
            subdataset_list=['train','test','validation'] #!
            
            from PIL import Image
            def imresize(arr, resize_list):
                return np.array(Image.fromarray(arr).resize((resize_list[0], resize_list[1])))
            def center_crop(x, crop_h=148, crop_w=148, resize_w=64):
                # crop the images to [crop_h,crop_w,3] then resize to [resize_h,resize_w,3]
                if crop_w is None:
                    crop_w = crop_h # the width and height after cropped
                h, w = x.shape[:2]
                j = int(round((h - crop_h)/2.))
                i = int(round((w - crop_w)/2.))
                return imresize(x[j:j+crop_h, i:i+crop_w], [resize_w, resize_w]) 
            def preprocess(batch):
                batch_new=[]
                for img in batch:
                    batch_new.append(center_crop(img).astype(np.float32)/255.0)
                batch=np.stack(batch_new, axis=0)
                return batch
            
            os.environ['CUDA_VISIBLE_DEVICES']=''
            import tensorflow_datasets as tfds
            #Load/preprocess/save train data
            for subdataset in subdataset_list:
                ds=tfds.load('celeb_a', split=subdataset, with_info=False, batch_size=-1)
                ds=tfds.as_numpy(ds)
                for key in key_list:
                    ds_key=ds[key]
                    if key=='image':
                        ds_key=preprocess(ds_key)
                    print(str(key))
                    np.save(data_dir+str(subdataset)+'_'+str(key)+'.npy', ds_key)
        
        #Load data from npy file
        if not analysis_mode:
            train_images=np.load(data_dir+'train_image.npy')
            validation_images=np.load(data_dir+'train_image.npy')
            TRAIN_BUF = train_images.shape[0]
        
        test_images=np.load(data_dir+'test_image.npy')
        image_shape=test_images.shape[1:] 
        
        TEST_BUF =  test_images.shape[0]
        np.random.seed(test_dataset_seed)
        test_inds=np.arange(TEST_BUF)
        np.random.shuffle(test_inds)
        test_images=test_images[test_inds]
        
        
        if analysis_mode:
            test_labels_list=[]
            test_attributes=np.load(data_dir+'test_attributes.npy', allow_pickle=True).item()
            attributes_key_list=list(test_attributes.keys())
            for key in attributes_key_list:
                test_labels_list.append(test_attributes[key].astype(np.float32))
            test_landmarks=np.load(data_dir+'test_landmarks.npy', allow_pickle=True).item()
            landmarks_key_list=list(test_landmarks.keys())
            for key in landmarks_key_list:
                test_labels_list.append(test_landmarks[key].astype(np.float32))
            
            test_labels=np.stack(test_labels_list, axis=1)
            test_labels=test_labels[test_inds]
            print('Keys: ')
            for i,key in enumerate(attributes_key_list+landmarks_key_list):
                print(i, key)
            test_dataset_with_labels = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)
            return test_dataset_with_labels, image_shape, attributes_key_list+landmarks_key_list
        
        if validation>0:
            train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(batch_size)
            validation_dataset = tf.data.Dataset.from_tensor_slices(validation_images).batch(batch_size)
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)
        
        if logP_batch_size>0:
            test_dataset_for_logP=tf.data.Dataset.from_tensor_slices(test_images).batch(logP_batch_size)
        else:
            test_dataset_for_logP=[]
        if validation>0:
            return train_dataset, test_dataset, test_dataset_for_logP, image_shape, validation_dataset
        else:
            return train_dataset, test_dataset, test_dataset_for_logP, image_shape
    
    elif dataset=='dsprites':
        from copy import deepcopy as copy
        
        def standard_shape(mode, xy):
            xy=copy(xy)
            if mode==1:
                #Square with a=b=1
                return (xy[:,:,0]<0.5)*(xy[:,:,0]>-0.5)*(xy[:,:,1]<0.5)*(xy[:,:,1]>-0.5)
            if mode==2:
                #ellipse with a=(2/np.pi)**0.5 b=0.5*a
                r2=2/np.pi
                return xy[:,:,0]**2+xy[:,:,1]**2*4<r2
            if mode==3:
                #Triangle 
                a=2**0.5*3**(-1)
                h1=xy[:,:,1]+a
                angle=2*np.pi/3
                matrix=np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]]).astype(np.float32)
        
                xy=np.dot(xy, matrix)
                h2=xy[:,:,1]+a
                xy=np.dot(xy, matrix)
                h3=xy[:,:,1]+a
                return (h1>=0)*(h2>=0)*(h3>=0)
        def shape(mode, xy, center, size, angle):
            xy=xy-center
            matrix=np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]]).astype(np.float32)
            xy=np.dot(xy, matrix)
            xy=xy/size
            return standard_shape(mode, xy)

        base=np.zeros([64,64,2])
        for i in range(base.shape[0]):
            for j in range(base.shape[1]):
                base[i][j][0]=i
                base[i][j][1]=j
        
        shape_array=np.arange(1,4,1)        
        pos_x_array=np.arange(20, 46, 2)
        pos_y_array=np.arange(20, 46, 2)
        size_array=np.arange(10, 21, 1)
        angle_array=np.arange(0, 2*np.pi, 2*np.pi/20)

        def generate_fig(batch_size=100):
            shape_batch=np.random.choice(shape_array, batch_size)
            pos_x_batch=np.random.choice(pos_x_array, batch_size)
            pos_y_batch=np.random.choice(pos_y_array, batch_size)
            
            pos_y_batch=(pos_x_batch+pos_y_batch)/2 #Introduce correaltion
            
            pos_batch=np.stack([pos_x_batch,pos_y_batch], axis=1)
            size_batch=np.random.choice(size_array, batch_size)
            angle_batch=np.random.choice(angle_array, batch_size)
            fig_list=[shape(shape_batch[i], base, pos_batch[i], size_batch[i], angle_batch[i]) for i in range(batch_size)]
            info=[pos_x_batch, pos_y_batch, shape_batch, size_batch, angle_batch]
            info=np.stack(info, axis=1).astype(np.float32)
            out=np.stack(fig_list, axis=0).astype(np.float32)
            out=np.expand_dims(out, axis=-1)
            return out, info

        image_shape=[64,64,1]
        
        if analysis_mode:
            print('Keys:')
            for i, key in enumerate(['pos_x', 'pos_y', 'shape', 'size', 'angle']):
                print(i, key)
            test_dataset_with_labels=[generate_fig(batch_size) for i in range(2000//batch_size)]
            return test_dataset_with_labels, image_shape
        
        train_dataset=[generate_fig(batch_size)[0] for i in range(50000//batch_size)]
        test_dataset=[generate_fig(batch_size)[0] for i in range(10000//batch_size)]
        test_dataset_for_logP=[generate_fig(logP_batch_size)[0] for i in range(10000//logP_batch_size)]
        
        
        return train_dataset, test_dataset, test_dataset_for_logP, image_shape
    

    
if __name__=='__main__':
    import os
    from time import time
    os.environ['CUDA_VISIBLE_DEVICES']=''
    train_dataset, test_dataset, test_dataset_for_logP, image_shape, validation_dataset=load_dataset('mnist', labels_for_mnist=[1], batch_size=10)
        