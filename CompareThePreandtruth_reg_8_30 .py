#%%
# design for multi-examples (2017_6_6)
#%%

import sys
import os
import subprocess
import numpy as np # Numpy for general purpose processing
import SimpleITK as sitk # SimpleITK to load images
from sklearn.metrics import f1_score

sys.path.insert(0, '../')
import DataPreProcessing.Func_data_preprocessing as Dpp

#%%
OrganPatch = True
LocalPatach = False
#organ = '187_gallbladder'
organ = '58_liver'
constraintPredict = 0.3
constraintgroundtruth = 0.02

#%%
# predict dir

root_dir = '/media/louis/Volume/ProgramWorkResult/MICCAI_2017_6_12/Result_Eigenfaces_KNN/RegPatchedImage'
if OrganPatch == True:
    result_dir_pre   = root_dir + '/'+organ + '/PCA_predict'
elif LocalPatach == True:
    result_dir_pre   = root_dir + '/'+organ + 'LocalPatch'+ '/PCA_predict'
else:
    raise ValueError('please choose the patch pattern')        

print 'Results will be stored in:', result_dir_pre
if not os.path.exists(result_dir_pre):
    subprocess.call('mkdir ' + '-p ' + result_dir_pre, shell=True)

#--------------- --------------------------------------------------------------
im_fns = Dpp.readTxtIntoList(result_dir_pre + '/' + 'FileList.txt')
predict_list = Dpp.ImageModificationAfterTransform(im_fns, result_dir_pre, constraintPredict)

# ground truth dir

root_dir = '/media/louis/Volume/ProgramWorkResult/MICCAI_2017_6_12/Result_Eigenfaces_KNN/RegPatchedImage'
if OrganPatch == True:
    result_dir_true   = root_dir + '/'+organ + '/PCA_test'
elif LocalPatach == True:
    result_dir_true   = root_dir + '/'+organ + 'LocalPatch'+ '/PCA_test'
else:
    raise ValueError('please choose the patch pattern')  

if not os.path.exists(result_dir_true):
    subprocess.call('mkdir ' + '-p ' + result_dir_true, shell=True)

#------------------------------------------------------------------------------
im_fns = Dpp.readTxtIntoList(result_dir_true + '/' + 'FileList.txt')
groundTruth_list = Dpp.ImageModificationAfterTransform(im_fns, result_dir_true, constraintgroundtruth)

#------------------------------------------------------------------------------
NN = len(im_fns)
diceScore = np.zeros((NN,))    
for i in xrange(NN):
    
    ImgroundTruth = sitk.ReadImage(groundTruth_list[i])
    TmGT_array = sitk.GetArrayFromImage(ImgroundTruth)
    y_true = np.reshape(TmGT_array,-1)
    

    ImPred = sitk.ReadImage(predict_list[i])
    ImPred_array = sitk.GetArrayFromImage(ImPred)
    y_pred = np.reshape(ImPred_array,-1)
    
    diceScore[i] = f1_score(y_true, y_pred)

dice_Statistics = {}
dice_Statistics['mean'] = np.mean(diceScore)
dice_Statistics['std'] = np.std(diceScore)
dice_Statistics['max'] = np.amax(diceScore)
dice_Statistics['min'] = np.amin(diceScore)
print dice_Statistics