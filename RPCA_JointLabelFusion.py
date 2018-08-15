__status__  = "Development"

'''
# Image segementation via KNN method in Eigenspace (organ level patched image)
# 03/06/2017
# Louis
# version 2
# a bug saveImagesFromDM testN must be 1. can use dicktionary later.
'''
#import resource
import sys
import os
import subprocess
import numpy as np
import copy
import string

sys.path.insert(0, '../')
import DataPreProcessing.Func_data_preprocessing as Dpp
import EigenfaceKNN.Func_Eigenface_KNN as EigKNN
import EigenfaceKNN.Func_k_nearest_neighbor_sklearning as KNNSklearn
from jointLabelFusion import MultiAtlasJointLabelFusion
    
#%%
#######################################  main #################################
'''# the reference iamges is used for determine the image origin, direction and spacing.''' 
regIfLinear = 'Linear'
#regIfLinear = 'nonlinear'

organ = '187_gallbladder'
#organ = '170_pancreas' 
#organ = '30325_left_adrenal_gland' 
#organ = '30324_right_adrenal_gland'
#organ = '29193_first_lumbar_vertebra' 

root_dir = '/media/data/louis/ProgramWorkResult/VisercialMAS_ANT' 
root_image_dir = os.path.join(root_dir, 'Patched_Image', 'GC_Volumes_patch_ANT')
root_label_dir = os.path.join(root_dir, 'Patched_Image', 'GC_label_patch_ANT')

result_root_dir = os.path.join(root_dir, 'Result_Eigenfaces_KNN')
result_dir = os.path.join(result_root_dir, 'PatchedImage', organ + '_'+ regIfLinear)

print 'Results will be stored in:', result_dir
if not os.path.exists(result_dir):
    subprocess.call('mkdir ' + '-p ' + result_dir, shell=True)   
    
outputPrefix = os.path.join(result_dir, 'JLF_predict/') 
if not os.path.exists(outputPrefix):
    subprocess.call('mkdir ' + '-p ' + outputPrefix, shell=True)  

# For reproducibility: save all parameters into the result dir
currentPyFile = os.path.realpath(__file__)
os.system('cp ' + currentPyFile+ ' ' +result_dir)

#%%
X_data_dir = root_image_dir + '/'+ organ + '_'+ regIfLinear+ '_Imagepatch' 
Y_data_dir = root_label_dir +'/'+ organ + '_'+ regIfLinear+ '_Labelpatch'
X_data_list = Dpp.readTxtIntoList(X_data_dir + '/FileList.txt')
Y_data_list = Dpp.readTxtIntoList(Y_data_dir + '/FileList.txt')
reference_im_fn = X_data_list[0]

# lamada: the tunning paramter that weights between the low-rank component and the sparse component
global lamda 
lamda = 0.7
tol = 1e-07

# train data and test data
data_num = len(X_data_list)
PcNo = data_num-1
MonteTime = 1
best_k = 30

#%%
outputPrefixList = []
for testN in xrange(data_num):
    
    selectedNO = range(data_num)
    train_selected = copy.copy(selectedNO)
    del train_selected[testN]
    test_selected = [selectedNO[testN]]    
    
    # training images
    im_file_list_train_X=[]
    for i in train_selected:
        im_file_list_train_X.append(X_data_list[i])
        
    # image for making eigenface basis
    im_file_list_Eigenface_basis = []
    im_file_list_Eigenface_basis = copy.copy(im_file_list_train_X)
   
    # computing the eigfaces
    Eigenface_basis_matrix = EigKNN.Imagelist_to_Matrix(im_file_list_Eigenface_basis)
    eigfaces, image_mean, low_rank, sparse = EigKNN.Eigenface_basis_RPCA(Eigenface_basis_matrix, PcNo, lamda,tol)
    del Eigenface_basis_matrix, sparse
    
    #%%
    # X_train
    train_image_matrix = EigKNN.Imagelist_to_Matrix(im_file_list_train_X)
    x_train = EigKNN.Eigenface_feature(train_image_matrix, image_mean, eigfaces)
    del train_image_matrix
    
    # Y_train
    im_file_list_train_Y=[]
    for i in train_selected:
        im_file_list_train_Y.append(Y_data_list[i])
    y_train = EigKNN.Imagelist_to_Matrix(im_file_list_train_Y)
    
    x_train = x_train.T # N*D
    y_train = y_train.T
    
    # X_test
    im_file_list_test_X=[]
    for i in test_selected:
        im_file_list_test_X.append(X_data_list[i])
        
    test_image_matrix = EigKNN.Imagelist_to_Matrix(im_file_list_test_X)
    x_test = EigKNN.Eigenface_feature(test_image_matrix, image_mean, eigfaces)
    del test_image_matrix   
   
    x_test = x_test.T
    
    # Y_test
    im_file_list_test_Y=[]
    for i in test_selected:
        im_file_list_test_Y.append(Y_data_list[i])  
        
#%%    
    classifier = KNNSklearn.KNearestNeighbor(algorithm='auto',metric='minkowski', p=2)
    classifier.train(x_train, y_train)
    _, selection = classifier.predict(x_test, k=best_k)    

    name, ext = os.path.splitext(im_file_list_test_Y[0])
    BaseName = os.path.basename(name)
    testIndicator = string.join(BaseName.split("_")[-6:-2], "_")
    outputfile = outputPrefix + testIndicator+'_posterior0001.nii.gz'
    outputPrefixList.append(outputfile)    
   
    currentSelection = list(selection.reshape(-1))
    atlasImageList = list(np.array(im_file_list_train_X)[currentSelection])
    atlasSegmentationList = list(np.array(im_file_list_train_Y)[currentSelection])
    TargetImageList = im_file_list_test_X 

    cmd = MultiAtlasJointLabelFusion(atlasImageList, atlasSegmentationList, TargetImageList, outputfile, EXECUTE = True)
    print('TestNo:'+str(testN))    
Dpp.WriteListtoFile(outputPrefixList, outputPrefix+"/FileList.txt")