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
regIfLinear = 'Linear'
#regIfLinear = 'nonlinear'
modility = 'wb'
#modility = 'CTce'
#modility = None

#organ = '29193_first_lumbar_vertebra' 
#organ = '170_pancreas' 
#constraintPredict = 0.2
organ = '187_gallbladder'
constraintPredict = 0.1
#organ = '30325_left_adrenal_gland' 
#constraintPredict = 0.2
#organ = '30324_right_adrenal_gland'
#constraintPredict = 0.2

constraintgroundtruth = 0.02

#%%
def DiceScoreCalculation(A,B):
    # 05/12/2016
    # louis
    # version 1.1
    # fouction: computing the dicescore
    
    k = 1
    constraint = 10**(-2)
    Nonzero_A = np.transpose(np.nonzero(np.abs(A) > constraint))
    Nonzero_B = np.transpose(np.nonzero(np.abs(B) > constraint))
    A[Nonzero_A]=k
    B[Nonzero_B]=k
    dice = np.sum(A[B==k])*2.0 / (np.sum(A) + np.sum(B))
    return dice  

def TPRCalculation(y_true,y_pred):
    # 05/12/2016
    # louis
    # version 1.1
    # fouction: computing the dicescore
    
    k = 1
    constraint = 10**(-2)
    Nonzero_A = np.transpose(np.nonzero(np.abs(y_true) > constraint))
    Nonzero_B = np.transpose(np.nonzero(np.abs(y_pred) > constraint))
    y_true[Nonzero_A]=k
    y_pred[Nonzero_B]=k
    dice = np.sum(y_true[y_pred==k])/(np.sum(y_true))
    return dice
  
def showDiceandTPR(groundTruth_list, predict_list):
    NN = len(groundTruth_list)
    diceScore = np.zeros((NN,))
    TPR = np.zeros((NN,))    
    for i in xrange(NN):
        
        ImgroundTruth = sitk.ReadImage(groundTruth_list[i])
        TmGT_array = sitk.GetArrayFromImage(ImgroundTruth)
        y_true = np.reshape(TmGT_array,-1)       
    
        ImPred = sitk.ReadImage(predict_list[i])
        ImPred_array = sitk.GetArrayFromImage(ImPred)
        y_pred = np.reshape(ImPred_array,-1)
        
        diceScore[i] = f1_score(y_true, y_pred)
        TPR[i] = TPRCalculation(y_true, y_pred)
        
    dice_Statistics = {}
    dice_Statistics['mean'] = np.mean(diceScore)
    dice_Statistics['std'] = np.std(diceScore)
    dice_Statistics['max'] = np.amax(diceScore)
    dice_Statistics['min'] = np.amin(diceScore)
    print'DICE:'
    print dice_Statistics
    
    TPR_Statistics = {}
    TPR_Statistics['mean'] = np.mean(TPR)
    TPR_Statistics['std'] = np.std(TPR)
    TPR_Statistics['max'] = np.amax(TPR)
    TPR_Statistics['min'] = np.amin(TPR)
    print'TPF'
    print TPR_Statistics
    
#%%
# predict dir

root_dir = '/media/data/louis/ProgramWorkResult/VisercialMAS_ANT/Result_Eigenfaces_KNN/PatchedImage'
result_dir_pre   = root_dir + '/'+organ + '_'+ regIfLinear + '/JLF_predict'
if not os.path.exists(result_dir_pre):
    subprocess.call('mkdir ' + '-p ' + result_dir_pre, shell=True)

#--------------------------------------------------------------------------
im_fns = Dpp.readTxtIntoList(result_dir_pre + '/' + 'FileList.txt')
predict_list = Dpp.ImageModificationAfterTransform(im_fns, result_dir_pre, constraintPredict)

RegOutput_dir   = root_dir + '/'+organ + '_'+ regIfLinear + '/JLF_predict'
Dpp.WriteListtoFile(predict_list, RegOutput_dir+"/FileListZoom.txt")   

# ground truth dir
root_dir = '/media/data/louis/ProgramWorkResult/VisercialMAS_ANT/Result_Eigenfaces_KNN/PatchedImage'
result_dir_true   = root_dir + '/'+organ + '_'+ regIfLinear + '/PCA_test'
if not os.path.exists(result_dir_true):
    subprocess.call('mkdir ' + '-p ' + result_dir_true, shell=True)

#--------------------------------------------------------------------------
im_fns = Dpp.readTxtIntoList(result_dir_true + '/' + 'FileList.txt')
groundTruth_list = Dpp.ImageModificationAfterTransform(im_fns, result_dir_true, constraintgroundtruth)

PREList= []
GTList= []
if modility != None:
    for item in predict_list:
        if modility in item:
            PREList.append(item)
    for item in groundTruth_list:
        if modility in item:
            GTList.append(item)
else:
    PREList= predict_list
    GTList= groundTruth_list
print organ
showDiceandTPR(GTList, PREList)
