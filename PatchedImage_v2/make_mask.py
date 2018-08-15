# -*- coding: utf-8 -*-

'''
# version 2.1
# louis
# 26.01.2017
# find the right patch for the image
'''
import sys
import os
import string
import subprocess
import numpy as np
sys.path.insert(0, '../')
import DataPreProcessing.Func_data_preprocessing as Dpp
import DataPreProcessing.Func_image_patch_v2 as Ip

#%%
root_dir = '/media/data/louis/ProgramWorkResult/VisercialMAS_ANT'
reference_dir = root_dir + '/Full_Image' + '/GC_Volumes_adjustment/referenceImage'
reference_im_fn = reference_dir + '/reference_WithoutNorm' + '.nrrd'

#%%
# patching images
regIfLinear = "Linear"
organ = '187_gallbladder'
factor = 4.0
    
label_data_dir = root_dir + '/Full_Image' + '/GC_label_adjustment'+'/'+organ +\
'/Modification_Regtrans_' + regIfLinear + '_ANT'  
iamge_data_dir = root_dir + '/Full_Image' + '/GC_Volumes_adjustment/Regtrans_' +\
regIfLinear + '_ANT'   

im_fns_label = Dpp.readTxtIntoList(label_data_dir + '/FileList.txt')
im_fns = Dpp.readTxtIntoList(iamge_data_dir + '/FileList.txt')
imname, imext = os.path.splitext(im_fns[0])
imageNameList = imname.split("_")[0:-4]

im_fns_image=[]
for i in xrange(len(im_fns_label)):
    name, ext = os.path.splitext(im_fns_label[i])
    labelBaseName = os.path.basename(name)
    imageBaseName = string.join(imageNameList+labelBaseName.split("_")[-6:-2], "_")
    im_fns_image.append(imageBaseName + imext)              

image_result_dir = root_dir + '/Patched_Image/GC_Volumes_patch' + '_ANT' 
label_result_dir = root_dir + '/Patched_Image/GC_label_patch' + '_ANT' 
result_dir_img = image_result_dir + '/'+ organ +'_ImageMask' 
result_dir_lab = label_result_dir+ '/' + organ + '_LabelMask'      

print 'Results will be stored in:', image_result_dir, label_result_dir
if not os.path.exists(result_dir_lab):
    subprocess.call('mkdir ' + '-p ' + result_dir_lab, shell=True)
if not os.path.exists(result_dir_img):
    subprocess.call('mkdir ' + '-p ' + result_dir_img, shell=True)
    
#----------------------------------------------------------------------------
central_point, dim_range_sup = Ip.patch_indice(im_fns_label)

print dim_range_sup
dim_range_sup = (factor*(-(np.floor((-dim_range_sup)/factor))))
dim_range_sup.astype(int)
print dim_range_sup
#%%

# make label mask
reference_image = reference_im_fn
result_list_lab = Ip.make_image_mask (im_fns_label, central_point, dim_range_sup,\
reference_image, result_dir_lab)
Dpp.WriteListtoFile(result_list_lab, result_dir_lab+"/FileList.txt")
Dpp.WriteListtoFile(im_fns_label, result_dir_lab+"/InputLabelList.txt")

# make image mask
reference_image = reference_im_fn
result_list_img = Ip.make_image_mask (im_fns_image, central_point, dim_range_sup,\
reference_image, result_dir_img)
Dpp.WriteListtoFile(result_list_img, result_dir_img+"/FileList.txt")
Dpp.WriteListtoFile(im_fns_image, result_dir_img+"/InputImageList.txt")