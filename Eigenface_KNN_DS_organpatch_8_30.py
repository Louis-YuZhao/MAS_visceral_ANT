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
import SimpleITK as sitk
import DataPreProcessing.Func_data_preprocessing as Dpp
import EigenfaceKNN.Func_Eigenface_KNN as EigKNN
import EigenfaceKNN.Func_k_nearest_neighbor_sklearning as KNNSklearn
    
#%%
#######################################  main #################################
'''# the reference iamges is used for determine the image origin, direction and spacing.''' 
regIfLinear = 'Linear'
#regIfLinear = 'nonlinear'

#organ = '187_gallbladder'
#organ = '170_pancreas' 
#organ = '30325_left_adrenal_gland' 
#organ = '30324_right_adrenal_gland'
organ = '29193_first_lumbar_vertebra' 
root_dir = '/media/data/louis/ProgramWorkResult/VisercialMAS_ANT' 
root_image_dir = root_dir + '/Patched_Image' + '/GC_Volumes_patch' + '_ANT'
root_label_dir = root_dir + '/Patched_Image' + '/GC_label_patch' + '_ANT'

result_root_dir = root_dir + '/Result_Eigenfaces_KNN'
result_dir = result_root_dir + '/PatchedImage/' + organ + '_'+ regIfLinear

print 'Results will be stored in:', result_dir
if not os.path.exists(result_dir):
    subprocess.call('mkdir ' + '-p ' + result_dir, shell=True)   

# For reproducibility: save all parameters into the result dir
currentPyFile = os.path.realpath(__file__)
os.system('cp ' + currentPyFile+ ' ' +result_dir)

#%%
X_data_dir = root_image_dir + '/'+ organ + '_'+ regIfLinear+ '_Imagepatch' 
Y_data_dir = root_label_dir +'/'+ organ + '_'+ regIfLinear+ '_Labelpatch'
#root_SC_dir = '/home/louis/Downloads/raw_data/visceral/SC_Volumes_downsample'
#X_sc_data_dir = root_SC_dir + '/affinetrans_DownSampleFactor_0.5_0.5_0.5'
X_data_list = Dpp.readTxtIntoList(X_data_dir + '/FileList.txt')
Y_data_list = Dpp.readTxtIntoList(Y_data_dir + '/FileList.txt')
reference_im_fn = X_data_list[0]

# lamada: the tunning paramter that weights between the low-rank component and the sparse component
global lamda 
lamda = 0.7
tol = 1e-07

# train data and test data
data_num = len(X_data_list)
train_num = data_num -1
test_num = 1
PcNo = train_num
MonteTime = 1
best_k = 10
dice_matrix = np.zeros((test_num, 1))
Weatherreceive = 'True'
#TrainImageforBasis = range(10)

#%%
im_ref = sitk.ReadImage(reference_im_fn)
im_ref_array = sitk.GetArrayFromImage(im_ref) # get numpy array
z_dim, x_dim, y_dim = im_ref_array.shape # get 3D volume shape
y_test_pred_matrix = np.zeros((z_dim*x_dim*y_dim, data_num))
y_test_matrix = np.zeros((z_dim*x_dim*y_dim, data_num))

test_selected_list = []
for testN in xrange(data_num):
    
    selectedNO = range(testN) + range(testN+1,data_num)
    selectedNO.append(testN)
    train_selected = selectedNO[0 : train_num]
    test_selected = selectedNO[train_num : train_num + test_num]    
    
    # training images
    im_file_list_train_X=[]
    for i in train_selected:
        im_file_list_train_X.append(X_data_list[i])
        
    # image for making eigenface basis
    im_file_list_Eigenface_basis = []
    im_file_list_Eigenface_basis = copy.copy(im_file_list_train_X)
    #for i in TrainImageforBasis:
    #    im_file_list_Eigenface_basis.append(X_sc_data_dir+ '/ImAffineResult_'+str(i)+ '.nrrd')
    
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
    
    # Y_test
    im_file_list_test_Y=[]
    for i in test_selected:
        im_file_list_test_Y.append(Y_data_list[i])        
    y_test = EigKNN.Imagelist_to_Matrix(im_file_list_test_Y)
    
    x_test = x_test.T
    y_test = y_test.T
    
    test_selected_list.append(im_file_list_test_Y[0]) # test_num should be 1
  
#%%    
    classifier = KNNSklearn.KNearestNeighbor(algorithm='auto',metric='minkowski', p=2)
    classifier.train(x_train, y_train)
    y_test_pred,selection = classifier.predict(x_test, k=best_k)  
    y_test_pred_matrix[:,testN] = y_test_pred[:,0]
    y_test_matrix[:,testN] = (y_test.T)[:,0]
 
#    selectionList = np.ndarray.tolist(selection[0])
#    
#    Testdir = im_file_list_test_X
#    Traindir = [im_file_list_train_X[i] for i in selectionList]
#    TestTrain_volume = Testdir+Traindir
#    
#    Testdir_label = im_file_list_test_Y
#    Traindir_label = [im_file_list_train_Y[i] for i in selectionList]
#    TestTrain_label = Testdir_label+Traindir_label
    
#    Matlab_dir = '/media/data/louis/Test_Random_Walk/MatLabPython'
#    Matlab_root = Matlab_dir + '/' + organ 
#    if not os.path.exists(Matlab_root):
#        subprocess.call('mkdir ' + '-p ' + Matlab_root, shell=True)   
#       
#    Matlab_volume = Matlab_root +'/volume_test_'+str(testN)+'.txt'
#    Matlab_label = Matlab_root +'/label_test_'+str(testN)+'.txt'
#    WriteListtoFile(TestTrain_volume, Matlab_volume)
#    WriteListtoFile(TestTrain_label, Matlab_label)    
    
    dice=[]
    for i in range(x_test.shape[0]):
        dice_temp = EigKNN.DiceScoreCalculation(y_test[i],(y_test_pred.T)[i])
        dice.append(dice_temp)
    
    dice_matrix[:,0] = np.array(dice)
    dice_matrix_mean = np.mean(dice_matrix,axis=1)    
    dice_Statistics = {}
    dice_Statistics['mean'] = np.mean(dice_matrix_mean)
    print dice_Statistics
    
#%%      
if Weatherreceive != False:           

    outputPrefix = result_dir + '/PCA_predict/' 
    if not os.path.exists(outputPrefix):
        subprocess.call('mkdir ' + '-p ' + outputPrefix, shell=True)    
    outputtest = result_dir + '/PCA_test/'
    if not os.path.exists(outputtest):
        subprocess.call('mkdir ' + '-p ' + outputtest, shell=True)
    
    outputPrefixList = []
    outputtestList = []
    for Dir_i in test_selected_list:
        name, ext = os.path.splitext(Dir_i)
        BaseName = os.path.basename(name)
        testIndicator = string.join(BaseName.split("_")[-6:-2], "_")
        outputPrefixList.append(outputPrefix + testIndicator + ext)
        outputtestList.append(outputtest + testIndicator + ext)
    
    reference_im_pred = reference_im_fn
    EigKNN.saveImagesFromDM(y_test_pred_matrix, outputPrefixList, reference_im_pred)
    Dpp.WriteListtoFile(outputPrefixList, outputPrefix+"/FileList.txt")
    EigKNN.saveImagesFromDM(y_test_matrix, outputtestList, reference_im_pred)
    Dpp.WriteListtoFile(outputtestList, outputtest+"/FileList.txt")