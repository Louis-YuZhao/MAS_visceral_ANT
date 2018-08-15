# conditional random field method for postprocessing
# version 1.0
# time: 2017/08/16
#%%

import sys
import os
import subprocess
import numpy as np # Numpy for general purpose processing
import SimpleITK as sitk # SimpleITK to load images
from sklearn.metrics import f1_score

sys.path.insert(0, '../')
sys.path.insert(0, '/home/louis/Documents/program_work/Python/DenseCRFWrapper_2017_7_23')
from DataPreProcessing.Func_data_preprocessing import readTxtIntoList, WriteListtoFile 
from DataPreProcessing.Func_data_preprocessing import ImageNormlization, ImageModificationAfterTransform
from denseinference import CRFProcessor

#%%
def crfImageModification(InputImageList, result_dir, threshold1, threshold2 = 10**(-2)):
    outputlist = []
    N = len(InputImageList)
    for i in range(N):        
        image = sitk.ReadImage(InputImageList[i])
        image_array = sitk.GetArrayFromImage(image) # get numpy array

        image_array[image_array > threshold1] = 0.7
        image_array[(image_array > threshold2) & (image_array <= threshold1)] = 0.3
        image_array[image_array <= threshold2] = 0

        img = sitk.GetImageFromArray(image_array)
        img.SetOrigin(image.GetOrigin())
        img.SetSpacing(image.GetSpacing())        
        img.SetDirection(image.GetDirection())
        
        name, ext = os.path.splitext(InputImageList[i])        
        baseName = os.path.basename(name)
        fn = result_dir + '/thrModify_' + baseName + '.nrrd'
        outputlist.append(fn)
        sitk.WriteImage(img,fn) 
    return outputlist
#%%
OrganPatch = False
LocalPatach = True
organ = '187_gallbladder' 
constraint1 = 0.3
constraintgroundtruth = 0.02

#%%
#raw image
root = '/media/louis/Volume/ProgramWorkResult/MICCAI_2017_6_12'
image_dir = root + '/Patched_Image/GC_Volumes_patch'+'/'+organ + '_Imagepatch'
im_fns = readTxtIntoList(image_dir + '/' + 'FileList.txt')
normalizedImage_list = ImageNormlization(im_fns, image_dir)
WriteListtoFile(normalizedImage_list, image_dir+'/normalizedFileList.txt')
#%%
# predict dir

root_dir = '/media/louis/Volume/ProgramWorkResult/MICCAI_2017_6_12/Result_Eigenfaces_KNN/PatchedImage'
if OrganPatch == True:
    result_dir_pre   = root_dir + '/'+organ + '/PCA_predict'
elif LocalPatach == True:
    result_dir_pre   = root_dir + '/'+organ + 'LocalPatch'+ '/PCA_predict'
else:
    raise ValueError('please choose the patch pattern')        

print 'Results will be stored in:', result_dir_pre
if not os.path.exists(result_dir_pre):
    subprocess.call('mkdir ' + '-p ' + result_dir_pre, shell=True)

#--------------------------------------------------------------------------
im_fns = readTxtIntoList(result_dir_pre + '/' + 'FileList.txt')
#crfModify_list = crfImageModification(im_fns, result_dir_pre, threshold1 = 0.3,threshold2 = 10**(-2))
#WriteListtoFile(crfModify_list, result_dir_pre+"/crfModify.txt")
crfModify_list = im_fns

# ground truth dir

root_dir = '/media/louis/Volume/ProgramWorkResult/MICCAI_2017_6_12/Result_Eigenfaces_KNN/PatchedImage'
if OrganPatch == True:
    result_dir_true   = root_dir + '/'+organ + '/PCA_test'
elif LocalPatach == True:
    result_dir_true   = root_dir + '/'+organ + 'LocalPatch'+ '/PCA_test'
else:
    raise ValueError('please choose the patch pattern')  

if not os.path.exists(result_dir_true):
    subprocess.call('mkdir ' + '-p ' + result_dir_true, shell=True)

#--------------------------------------------------------------------------
im_fns = readTxtIntoList(result_dir_true + '/' + 'FileList.txt')
groundTruth_list = ImageModificationAfterTransform(im_fns, result_dir_true, constraintgroundtruth)
WriteListtoFile(groundTruth_list, result_dir_true + "/crfModify.txt")

#--------------------------------------------------------------------------
NN = len(im_fns)
diceScore = np.zeros((NN,))    
pro = CRFProcessor.CRF3DProcessor(max_iterations=30,
                 pos_x_std=0.508,
                 pos_y_std=0.508,
                 pos_z_std=0.508,
                 pos_w=43.0,
                 bilateral_x_std=1.3,
                 bilateral_y_std=1.3,
                 bilateral_z_std=1.3,
                 bilateral_intensity_std=0.0166,
                 bilateral_w=51.066,
                 dynamic_z=False,
                 ignore_memory=False,
                 verbose=False)

for i in xrange(NN):
        
    ImgroundTruth = sitk.ReadImage(groundTruth_list[i])
    TmGT_array = sitk.GetArrayFromImage(ImgroundTruth)
    y_true = np.reshape(TmGT_array,-1)
    
    rawimage = sitk.ReadImage(normalizedImage_list[i])
    rawimage_array = sitk.GetArrayFromImage(rawimage)
    
    ImPred = sitk.ReadImage(crfModify_list[i])
    ImPred_array = sitk.GetArrayFromImage(ImPred)
    probs = np.concatenate([1 - ImPred_array[..., None], ImPred_array[..., None]], axis=-1)
    
    crfImPred = pro.set_data_and_run(rawimage_array, probs)
    y_pred = np.reshape(crfImPred,-1)

    diceScore[i] = f1_score(y_true, y_pred)

dice_Statistics = {}
dice_Statistics['mean'] = np.mean(diceScore)
dice_Statistics['std'] = np.std(diceScore)
dice_Statistics['max'] = np.amax(diceScore)
dice_Statistics['min'] = np.amin(diceScore)
print dice_Statistics