__status__  = "Development"

'''
# 12/01/2017
# version: 1.0
# Functions for atlas_selection
# reference: Serag, Ahmed, et al. "A sparsity-based atlas selection technique for
  multiple-atlas segmentation: Application to neonatal brain labeling." Signal 
  Processing and Communication Application Conference (SIU), 2016 24th. IEEE, 2016.
# author: Yu ZHAO

'''

#%%
import numpy as np

#%%
def distance_canculating(imageA,imageB):
    '''
    # calculating the diatance between two image
    # input
      imageA: vectorized image A  
      imageB: vectorized image B
    # output
      distance: the distance between two images
      
    '''    
    distance = np.zeros_like(imageA)
    distance = np.linalg.norm(imageA - imageB, 2)
    distance = distance**2
    
    return distance
    
def distance_canculating_matrix(InputImages,RefImage):
    '''
    # calculating the diatance between two image
    # input
      InputImages: input vectorized images  
      RefImage: reference image
    # output
      distance: the distance between InputImages and RefImage
      
    '''
    temp = np.abs(InputImages - RefImage)
    temp = temp**2
    distance = np.sum(temp,axis = 0)    
   
    return distance

def image_select(InputImage, SelImageNum):
    '''
    # Images Selection
    # input 
      InputImage: the input images (M*N, N is the number of the images)
      SelImageNum: the number of selected number
    
    # output
      OutputImage: the selected images
    '''
    X = InputImage
    M,N = np.shape(X)
    
    assert N > SelImageNum, 'SelImageNum should be not greater than N'
        
    SelImage = np.zeros((M,SelImageNum))
    
    X_mean = np.mean(X, axis=1) 
    X_mean = X_mean[:, np.newaxis]
    dist1 = distance_canculating_matrix(X, X_mean)
    
    SelArg = np.argmin(dist1)
    SelImage[:,0] = X[:, SelArg] # the first selected S, S1
    RestArg = range(SelArg) + range(SelArg+1,N)    
    X_rest = X[:,RestArg]
    
    csin = 1 # current selected image number.
    while csin < SelImageNum:
        CurrResImageNO = X_rest.shape[1]
        d_is = np.zeros(CurrResImageNO)
        for i in range(CurrResImageNO):
            X_rest_i = X_rest[:,i]
            X_rest_i = X_rest_i[:,np.newaxis]
            dist = distance_canculating_matrix(SelImage[:,0:csin], X_rest_i)
            d_is[i] = np.mean(dist)
        
        SelArg = np.argmin(d_is)
        SelImage[:,csin] = X[:, SelArg]
        RestArg = range(SelArg) + range(SelArg+1,CurrResImageNO)    
        X_rest = X_rest[:,RestArg]
        csin += 1
        
    return SelImage
    
def image_select_one(InputImage):
    '''
    # Images Selection
    # input 
      InputImage: the input images (M*N, N is the number of the images)
      SelImageNum: the number of selected number
    
    # output
      OutputImage: the selected images
    '''
    X = InputImage
    M,N = np.shape(X)    
          
    SelImage = np.zeros((M,1))
    
    X_mean = np.mean(X, axis=1) 
    X_mean = X_mean[:, np.newaxis]
    dist1 = distance_canculating_matrix(X, X_mean)
    
    SelArg = np.argmin(dist1)
    SelImage[:,0] = X[:, SelArg]         
    return SelImage, SelArg