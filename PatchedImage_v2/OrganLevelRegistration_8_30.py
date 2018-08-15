# Organ level Registration
'''
2017.08.30
version 1
Author: Yu ZHAO

'''
import SimpleITK as sitk # SimpleITK to load images
import subprocess
import os
import sys
import string
import time
sys.path.insert(0, '../')
import DataPreProcessing.Func_data_preprocessing as Dpp

runImageRegistration = True

IFNormalization = False
#%%
#organ = '187_gallbladder'
organ = '58_liver'

#%%
root_dir = '/media/data/louis/ProgramWorkResult/VisercialMAS_ANT'
Input_Root_dir = root_dir + '/Patched_Image'
Image_Root_Dir = Input_Root_dir + '/GC_Volumes_patch_Elastix/' + organ + '_ImageMask'
Label_Root_Dir = Input_Root_dir + '/GC_label_patch_Elastix/' + organ + '_LabelMask'

Output_Root_dir = '/media/louis/Volume/ProgramWorkResult/MICCAI_2017_6_12/OrganReg_Elastix'
Image_Result_Dir = Output_Root_dir + '/Volume_OrganReg'+ '/' + organ
Label_Result_Dir = Output_Root_dir + '/Label_OrganReg'+ '/' + organ

# Make reference
reference_dir = root_dir + '/Full_Image/GC_Volumes_adjustment' + '/referenceImage'
        
selName = "10000128_1_CTce_ThAb"
DirFile = Dpp.readTxtIntoList(Image_Root_Dir+'/FileList.txt')
for dir_i in DirFile:
    if dir_i.find(selName) != -1:
        referenceImageMask = dir_i
        break            
#%% 
#  Registration
if IFNormalization != False: 
    reference_im_fn = reference_dir + '/reference_WithNorm' + '.nrrd'
else:
    reference_im_fn = reference_dir + '/reference_WithoutNorm' + '.nrrd'

Imagereginput_dir = Image_Root_Dir
images = Dpp.readTxtIntoList(Imagereginput_dir+'/InputImageList.txt')
iamges_mask = Dpp.readTxtIntoList(Imagereginput_dir+'/FileList.txt')
ImageRegOutput_dir = Image_Result_Dir
if not os.path.exists(ImageRegOutput_dir):
    subprocess.call('mkdir ' + '-p ' + ImageRegOutput_dir, shell=True)   

labelreginput_dir = Label_Root_Dir
labels = Dpp.readTxtIntoList(labelreginput_dir+'/InputLabelList.txt')
labels_mask = Dpp.readTxtIntoList(labelreginput_dir+'/FileList.txt')
LabelRegOutput_dir = Label_Result_Dir
if not os.path.exists(LabelRegOutput_dir):
    subprocess.call('mkdir ' + '-p ' + LabelRegOutput_dir, shell=True)
     
#------------------------------------------------------------------------------
     
if runImageRegistration != False:     
    s = time.time()   
    
    logFileDir = Output_Root_dir+'/RegTrans' + '_RUN_' + '.log'
    logFile = open(logFileDir, 'w')    
   
    fixedIm =  reference_im_fn    
    fixedImage = sitk.ReadImage(fixedIm)
    
    selx = sitk.ElastixImageFilter()
    selx.SetFixedImage(fixedImage)
    fixedImMask = referenceImageMask
    fixedImageMask = sitk.ReadImage(fixedImMask)
    fixedImageMask = sitk.Cast(fixedImageMask, sitk.sitkUInt8)
    selx.SetFixedMask(fixedImageMask)
    
    transformParaMaplist = []
    RegImOutputList = []
    
    for i in range(len(images)):    
        
        movingIm = images[i]        
        name, ext = os.path.splitext(movingIm)
        baseName = os.path.basename(name)
        imageBaseName = string.join(baseName.split("_")[-4:], "_")
        transformPara = ImageRegOutput_dir+'/ImRegPre_' + imageBaseName
        transformParaMaplist.append(transformPara)
        if not os.path.exists(transformPara):
            subprocess.call('mkdir ' + '-p ' + transformPara, shell=True)
        
        movingImage = sitk.ReadImage(movingIm)
        selx.SetMovingImage(movingImage)
        selx.SetOutputDirectory(transformPara)
        selx.Execute()

        # write the result image to the file 
        RegImOutput= ImageRegOutput_dir+'/ImRegResult_' + imageBaseName + '.nrrd'
        RegImOutputList.append(RegImOutput)
        sitk.WriteImage(selx.GetResultImage(), RegImOutput)
        print ("the %d th registration is finished" % (i))   

#%%    
    stran = sitk.TransformixImageFilter()
    RegLabOutputList = []
  
    for j in range(len(images)):             
                
        movingIm = labels[j] 
        movingImage = sitk.ReadImage(movingIm)
                
        transformParadir_Trans = transformParaMaplist[j] + '/TransformParameters.0.txt'
        paraTrans = sitk.ReadParameterFile(transformParadir_Trans)
        paraTrans['FinalBSplineInterpolationOrder'] = ['0']
        
        transformParadir_Affine = transformParaMaplist[j] + '/TransformParameters.1.txt' 
        paraAffine = sitk.ReadParameterFile(transformParadir_Affine)
        paraAffine['FinalBSplineInterpolationOrder'] = ['0']            
       
        stran.SetMovingImage(movingImage)          
        stran.SetTransformParameterMap(paraTrans)
        stran.AddTransformParameterMap(paraAffine)
        stran.SetOutputDirectory(LabelRegOutput_dir)
        stran.Execute()        

        # write the result image to the file 
        name, ext = os.path.splitext(movingIm)
        baseName = os.path.basename(name)
        labelBaseName = string.join(baseName.split("_")[-6:], "_")
        RegLabOutput = LabelRegOutput_dir + '/LabelRegResult_' + labelBaseName + '.nrrd'
        RegLabOutputList.append(RegLabOutput)
        sitk.WriteImage(stran.GetResultImage(), RegLabOutput)
        print ("the %d th registration is finished" % (j))  
           
    e = time.time()
    l = e - s
    
    Dpp.WriteListtoFile(RegImOutputList, ImageRegOutput_dir+"/FileList.txt")
    Dpp.WriteListtoFile(RegLabOutputList, LabelRegOutput_dir+"/FileList.txt")
    print 'affine registration is finished'
    print 'Total running time: %f mins'%(l/60.0)