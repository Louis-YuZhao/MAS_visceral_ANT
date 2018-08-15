__status__  = "Development"

'''
# Image segementation via KNN method in Eigenspace (suborgan level patched image)
# 06/06/2017
# Louis
# version 1

'''
#import resource
import sys
import os
import numpy as np
import subprocess
import time
import string
import SimpleITK as sitk

sys.path.insert(0, '../')
import DataPreProcessing.Func_data_preprocessing as Dpp
import EigenfaceKNN.Func_Eigenface_KNN_patch_6_6 as EigKNNPatch    

#%%
####################################### main ###################################  
regIfLinear = 'Linear'
organ = '187_gallbladder' 
root_dir = '/media/louis/Volume/ProgramWorkResult/MICCAI_2017_6_12' 
root_image_dir = root_dir + '/Patched_Image' + '/GC_Volumes_patch' + '_Elastix'
root_label_dir = root_dir + '/Patched_Image' + '/GC_label_patch' + '_Elastix'

result_root_dir = root_dir + '/Result_Eigenfaces_KNN'
result_dir = result_root_dir + '/PatchedImage/' + + organ + '_'+ regIfLinear + '_LocalPatch'

print 'Results will be stored in:', result_dir
if not os.path.exists(result_dir):
    subprocess.call('mkdir ' + '-p ' + result_dir, shell=True)   

# For reproducibility: save all parameters into the result dir
currentPyFile = os.path.realpath(__file__)
os.system('cp' + currentPyFile+ ' ' +result_dir)

#%%
X_data_dir = root_image_dir + '/'+ organ + '_'+ regIfLinear+ '_Imagepatch_withOrganReg' 
Y_data_dir = root_label_dir +'/'+ organ + '_'+ regIfLinear+ '_Labelpatch_withOrganReg'

#root_SC_dir = '/home/louis/Downloads/raw_data/visceral/SC_Volumes_downsample'
#X_sc_data_dir = root_SC_dir + '/affinetrans_DownSampleFactor_0.5_0.5_0.5'

X_data_list = Dpp.readTxtIntoList(X_data_dir + '/FileList.txt')
Y_data_list = Dpp.readTxtIntoList(Y_data_dir + '/FileList.txt')
reference_im_fn = X_data_list[0]

im_ref = sitk.ReadImage(reference_im_fn)
image_array = sitk.GetArrayFromImage(im_ref) # get numpy array
z_dim, x_dim, y_dim = image_array.shape
print z_dim, x_dim, y_dim
# the reference iamges is used for determine the image origin, direction and spacing.

OriginalImPara = {}
OriginalImPara['Origin'] = im_ref.GetOrigin()
OriginalImPara['Spacing'] = im_ref.GetSpacing()
OriginalImPara['Direction'] = im_ref.GetDirection()
OriginalImPara['Dim'] = (z_dim, x_dim, y_dim)

InputDataDir = {}
InputDataDir ['X_data_list'] = X_data_list
InputDataDir ['Y_data_list'] = Y_data_list
#InputDataDir ['X_sc_data_dir'] = X_sc_data_dir
             
# train data and test data
KNNPara ={}
KNNPara['data_num'] = 36
KNNPara['train_num'] = 35
KNNPara['test_num'] = 1
KNNPara['PcNo'] = 35
KNNPara['K'] = 10
KNNPara['RandomChoice'] = False
KNNPara['MovePara'] = dict([['z_move', 0],['x_move', 0],['y_move', 0]])
Weatherreceive = True
#KNNPara['TrainImageforBasis'] = range(10)

#%%
# lamada: the tunning paramter that weights between the low-rank component and the sparse component
RPCA_Para = {}
RPCA_Para['lamda'] = 0.7
RPCA_Para['tol'] = 1e-07

pad_para = {}
pad_para['mode'] = 'constant'
pad_para['constant_values'] = 0

PatchSize = (6,6,6)
p_z = PatchSize[0]
p_x = PatchSize[1]
p_y = PatchSize[2]

if (z_dim % p_z) != 0:
    pad_z = (p_z - (z_dim % p_z))/2
else:
    pad_z = 0
    
if (x_dim % p_x) != 0:
    pad_x = (p_x - (x_dim % p_x))/2
else:
    pad_x = 0
    
if (y_dim % p_y) != 0:
    pad_y = (p_y - (y_dim % p_y))/2
else:
    pad_y = 0

pad_para['pad_width'] = ((pad_z,pad_z),(pad_x,pad_x),(pad_y,pad_y))

#%%
data_num = KNNPara['data_num']
y_test_pred_matrix = np.zeros((z_dim*x_dim*y_dim, data_num))
y_test_matrix = np.zeros((z_dim*x_dim*y_dim, data_num))

test_selected_list = []
for testN in xrange(data_num):
    
    selection = range(testN) + range(testN+1, data_num)
    selection.append(testN)
    KNNPara['ImageMake'] = selection
    
    Tbegin = time.time()    
    y_test_pred, y_test = EigKNNPatch.KNNPrediction_LocalPatch(InputDataDir,\
                    KNNPara, OriginalImPara, PatchSize, pad_para, RPCA_Para)
    Tend = time.time()
    print 'Time uses : %d seconds' %((Tend-Tbegin))
    y_test_pred_matrix[:, testN] = y_test_pred[:, 0]
    y_test_matrix[:, testN] = y_test[:, 0]

    test_selected_list.append(Y_data_list[testN]) # test_num should be 1    

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
    
    EigKNNPatch.saveImagesFromDataMatrix(y_test_pred_matrix, outputPrefixList, OriginalImPara)
    Dpp.WriteListtoFile(outputPrefixList, outputPrefix+"/FileList.txt")
    EigKNNPatch.saveImagesFromDataMatrix(y_test_matrix, outputtestList, OriginalImPara)
    Dpp.WriteListtoFile(outputtestList, outputtest+"/FileList.txt")