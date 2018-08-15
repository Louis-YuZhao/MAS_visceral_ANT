# conditional random field method for postprocessing
# fitting for selecting the best parameters
# version 1.0
# time: 2017/08/16
#%%

import sys
import os
import subprocess
import SimpleITK as sitk # SimpleITK to load images
from scipy.stats import expon

sys.path.insert(0, '../')
sys.path.insert(0, '/home/louis/Documents/program_work/Python/DenseCRFWrapper_2017_7_23')
from DataPreProcessing.Func_data_preprocessing import readTxtIntoList, WriteListtoFile 
from DataPreProcessing.Func_data_preprocessing import ImageNormlization, ImageModificationAfterTransform
from denseinference import CRFProcessorEsther

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
OrganPatch = True
LocalPatach = False
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
crfModify_list = crfImageModification(im_fns, result_dir_pre, threshold1 = 0.3,threshold2 = 10**(-2))
WriteListtoFile(crfModify_list, result_dir_pre+"/crfModify.txt")

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

all_images = []
all_probs = []
all_truths = []

for i in xrange(NN):
    
    rawimage = sitk.ReadImage(normalizedImage_list[i])
    rawimage_array = sitk.GetArrayFromImage(rawimage)
    all_images.append(rawimage_array)
    
    ImPred = sitk.ReadImage(crfModify_list[i])
    ImPred_array = sitk.GetArrayFromImage(ImPred)
    all_probs.append(ImPred_array)
    
    ImgroundTruth = sitk.ReadImage(groundTruth_list[i])
    TmGT_array = sitk.GetArrayFromImage(ImgroundTruth)
    all_truths.append(TmGT_array)

pro = CRFProcessorEsther.CRF3DProcessor(verbose=True)

#PARAMGRID = dict(max_iterations=[10, 20, ],
#                 pos_x_std=expon(scale=10, loc=0.01),
#                 #                  pos_y_std=expon(scale=5),
#                 #                  pos_z_std=expon(scale=5),
#
#                 bilateral_x_std=expon(scale=30, loc=0.01),
#                 #                  bilateral_y_std=expon(scale=30),
#                 #                  bilateral_z_std=expon(scale=30),
#
#                 # the intensities are scaled between 0 and 255 in cpp
#                 # we should pick a bilateral std of something that scales to this range 
#                 bilateral_intensity_std=expon(scale=.20, loc=0.01),
#
#                 # the weights should not be smaller than 1.
#                 pos_w=expon(scale=6, loc=1),
#                 bilateral_w=expon(scale=6, loc=1),)

PARAMGRID = dict(max_iterations=[20, 30],
                 pos_x_std=[0.508306177387],
                 #                  pos_y_std=expon(scale=5),
                 #                  pos_z_std=expon(scale=5),

                 bilateral_x_std=[1.86],
                 #                  bilateral_y_std=expon(scale=30),
                 #                  bilateral_z_std=expon(scale=30),

                 # the intensities are scaled between 0 and 255 in cpp
                 # we should pick a bilateral std of something that scales to this range 
                 bilateral_intensity_std=[1.86],

                 # the weights should not be smaller than 1.
                 pos_w=expon(scale=6, loc=1),
                 bilateral_w=expon(scale=6, loc=1),)

pro.fit(all_images, all_probs, all_truths, scorer=None, n_iter=1000, paramgrid=PARAMGRID,logfile=None)