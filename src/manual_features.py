#Designed for fashion_mnist and mnist
import numpy as np
from copy import deepcopy as copy

bright_threshold=0.05
def brightness(img):
    mask=img>bright_threshold
    img_bright_sum=np.sum(img*mask)
    img_mask_sum=np.sum(mask)
    if img_mask_sum<=0:
        return 0
    return img_bright_sum/img_mask_sum

width_threshold=0.1
def width(img):
    mask=img>bright_threshold
    row_width=np.mean(mask, axis=1)
    width_mask=row_width>width_threshold
    width_sum=np.sum(row_width*width_mask)
    width_mask_sum=np.sum(width_mask)
    if width_mask_sum<=0:
        return 0
    return width_sum/width_mask_sum

height_threshold=0.1
def height(img):
    mask=img>bright_threshold
    column_height=np.mean(mask, axis=1)
    height_mask=column_height>height_threshold
    height_sum=np.sum(column_height*height_mask)
    height_mask_sum=np.sum(height_mask)
    if height_mask_sum<=0:
        return 0
    return height_sum/height_mask_sum

def slope(x, y):
    #Not a feature
    x=x-x.mean()
    y=y-y.mean()
    return np.sum(x*y)/np.sum(x**2)

def angle_from_top(img):
    mask=img>bright_threshold
    row_width=np.mean(mask, axis=1)
    width_mask=row_width>width_threshold
    row_width_masked=[]
    for i in range(width_mask.shape[0]):
        if width_mask[i]:
            row_width_masked.append(row_width[i])
    row_width_masked=np.array(row_width_masked)
    if row_width_masked.shape[0]==0:
        return 0
    return slope(np.arange(row_width_masked.shape[0]), row_width_masked)

def angle_from_middle(img):
    mask=img>bright_threshold
    row_width=np.mean(mask, axis=1)
    width_mask=row_width>width_threshold
    row_width_masked=[]
    for i in range(width_mask.shape[0]):
        if width_mask[i]:
            row_width_masked.append(row_width[i])
    row_width_masked=np.array(row_width_masked)
    
    l=int(row_width_masked.shape[0])
    if l==0:
        return 0
    if l%2==0:
        x=np.concatenate([np.flip(np.arange(int(l/2))), np.arange(int(l/2))], axis=0)
    else:
        x=np.concatenate([np.flip(np.arange(1, int((l+1)/2))),np.array([0]), np.arange(1, int((l+1)/2))], axis=0)
    return slope(x, row_width_masked)

def angle(img):
    mask=img>bright_threshold
    row_width=np.mean(mask, axis=1)
    
    width_mask=row_width>width_threshold
    
    mid_point=[]
    row_axis=np.arange(img.shape[1])
    for i in range(width_mask.shape[0]):
        if width_mask[i]:
            mid_point.append(np.sum(row_axis*mask[i])/np.sum(mask[i]))
    mid_point=np.array(mid_point)
    
    return slope(np.arange(mid_point.shape[0]), mid_point)

feature_function=[brightness, height, width, angle_from_top, angle_from_middle, angle]

def feature(img):
    if img.ndim==2:
        feature_list=[function(img) for function in feature_function]
        return np.array(feature_list)
    elif img.ndim==3:
        feature_list=[feature(img_) for img_ in img]
        return np.stack(feature_list, axis=0)
    elif img.ndim==4:
        img=np.mean(img, axis=-1)
        return feature(img)

def weighting(features):
    mean=np.mean(features, axis=0, keepdims=True)
    std=np.std(features, axis=0, keepdims=True)
    return (features-mean)/(std+1e-7)

def class_weighting(labels, features):
    features_new=copy(features)
    labels=labels.reshape([-1])
    label_set=set(labels)
    for label in label_set:
        ind=[i for i in range(labels.shape[0]) if labels[i]==label]
        features_new[ind]=weighting(features_new[ind])
    return features_new
    
if __name__=='__main__':
    mode=1
    if mode==0:
        from src.dataset_utils import load_dataset

        batch_size=10
        dataset='mnist'
        test_dataset_with_labels, image_shape=load_dataset(dataset, batch_size, analysis_mode=True)

        for imgs, labels in test_dataset_with_labels:
            imgs=imgs
            break
        
        for i in range(1):
            print(feature(imgs))
            print([function.__name__ for function in feature_function])
        
    elif mode==1:
        labels=np.array([1,2,1,3,1,2,1,3]).reshape([-1, 1])
        features=np.array([[1,1],[-1,2],[1,2],[2,3],[1,3],[1,0],[1,4],[4,3]]).astype(np.float32)
        print(class_weighting(labels, features))
        #print(features[[0,2,4,6]], weighting(features[[0,2,4,6]]))
        
    
