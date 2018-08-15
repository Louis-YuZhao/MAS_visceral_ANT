# -*- coding: utf-8 -*-
# Louis
# 07/06/2017
# version 1
# 
# postprocessing for showing the final result

#%%
import numpy as np # Numpy for general purpose processing
import SimpleITK as sitk # SimpleITK to load images
import subprocess
import os
import scipy.ndimage
from DataPreProcessing.Func_data_preprocessing import readTxtIntoList, WriteListtoFile 

#%%
def imagezoom(InputImageList, output,samplefactor, InterP):
    num_data = len(InputImageList)
    outputList = []
    for i in range(num_data):        
        image = sitk.ReadImage(InputImageList[i])
        image_array = sitk.GetArrayFromImage(image) # get numpy array
        z_dim, x_dim, y_dim = image_array.shape 
        
        # image upsampering
#        im = scipy.ndimage.interpolation.zoom(image_array, samplefactor, order = InterP)          
        img = sitk.GetImageFromArray(image_array)   
        img.SetOrigin(image.GetOrigin())    
        img.SetSpacing(image.GetSpacing())        
        img.SetDirection(image.GetDirection())        
        
        name, ext = os.path.splitext(InputImageList[i])
        baseName = os.path.basename(name)
        fn= output+'_Imzoomed_' + baseName + '.nrrd'        
        sitk.WriteImage(img,fn)
        outputList.append(fn) 
        print 'Iteration Num %d is finished'% i        
    del image,image_array    
    return outputList

def predictRecover(predictlist, central_point, ImageInfo, outputdir):
    
    ImageDim = ImageInfo['dimAdj']
    z_im=ImageDim[0]; x_im=ImageDim[1]; y_im=ImageDim[2]
    num_data = len(predictlist)
    labelRecover = np.zeros(tuple(ImageDim))
    outputList = []
    for i in range(num_data):        
        image = sitk.ReadImage(predictlist[i])
        image_array = sitk.GetArrayFromImage(image) # get numpy array
        z_dim, x_dim, y_dim = image_array.shape 
        
        # recover
        z_begin=int(central_point[0,i]-(z_dim/2))
        x_begin=int(central_point[1,i]-(x_dim/2))
        y_begin=int(central_point[2,i]-(y_dim/2))
        
        z_end=z_begin+z_dim
        x_end=x_begin+x_dim
        y_end=y_begin+y_dim
        
        if z_begin < 0:
            pad_z = np.abs(0-z_begin)
            z_begin=z_begin+pad_z
            z_end=z_end+pad_z
        elif z_end > z_im:
            pad_z = np.abs(z_end-z_im)
            z_begin=z_begin-pad_z
            z_end=z_end-pad_z
        else:
            pass
        
        if x_begin < 0:
            pad_x = np.abs(0-x_begin)
            x_begin=x_begin+pad_x
            x_end=x_end+pad_x
        elif x_end > x_im:
            pad_x = np.abs(x_end-x_im)
            x_begin=x_begin-pad_x
            x_end=x_end-pad_x
        else:
            pass            
             
        if y_begin < 0:
            pad_y = np.abs(0-y_begin)
            y_begin=y_begin+pad_y
            y_end=y_end+pad_y
        elif y_end > y_im:
            pad_y = np.abs(y_end-y_im)
            y_begin=y_begin-pad_y
            y_end=y_end-pad_y
        else:
            pass
        
        labelRecover[z_begin:(z_begin+z_dim),x_begin:(x_begin+x_dim),y_begin:(y_begin+y_dim)]=image_array
        # write image       
        img = sitk.GetImageFromArray(labelRecover)   
        img.SetOrigin(ImageInfo['origin'])    
        img.SetSpacing(ImageInfo['spacing'])        
        img.SetDirection(ImageInfo['direction'])
        
        name, ext = os.path.splitext(predictlist[i])
        baseName = os.path.basename(name)
        fn = outputdir + '/predictRecover_' + baseName + '.nrrd'
        sitk.WriteImage(img,fn)
        outputList.append(fn) 
        print 'Iteration Num %d is finished'% i        
    del image,image_array
    return outputList        

#%%
regIfLinear = 'Linear'
#regIfLinear = 'nonlinear'    

organ = '187_gallbladder'
organ = '170_pancreas' 
organ = '30325_left_adrenal_gland' 
organ = '30324_right_adrenal_gland'
organ = '29193_first_lumbar_vertebra' 

CWD = os.getcwd()
central_point = np.load(CWD+'/PatchedImage_v2/'+ organ + '_'+ regIfLinear+'_central_point.npy')
ImageInfo = np.load(CWD+'/PreProcessing_v5/ImageInfo.npy')
ImageInfo = ImageInfo.item()

root = '/media/louis/Volume/ProgramWorkResult/VisercialMAS_ANT'
rawPredictDir = root + "/Result_Eigenfaces_KNN/PatchedImage/"+organ + '_'+ regIfLinear+"/PCA_predict"
rawPredictlist = readTxtIntoList(rawPredictDir + "/FileListZoom.txt")

PredictDir = root + "/PredictRecovery/"+organ + '_'+ regIfLinear+"_PredictRecovery"
if not os.path.exists(PredictDir):
    subprocess.call('mkdir ' + '-p ' + PredictDir, shell=True)
   
predict_List = predictRecover(rawPredictlist, central_point, ImageInfo, PredictDir)
WriteListtoFile(predict_List, PredictDir + "/FileList.txt")

#%%
# upsample

#root_dir = '/media/louis/Volume/ProgramWorkResult/MICCAI_2017_6_12/Full_Image'
#rawImageDir = root_dir + '/GC_Volumes_adjustment/Regtrans_linear'
#groundTruthDir = root_dir + '/GC_label_adjustment'+'/'+organ + '_'+ regIfLinear + '/Modification_Regtrans_linear'

resultDir = root + "/UpsampleResult/"+organ + '_'+ regIfLinear+"_UpsampleResult"
if not os.path.exists(resultDir):
    subprocess.call('mkdir ' + '-p ' + resultDir, shell=True)   
  
#rawImageList=readTxtIntoList(rawImageDir+"/FileList.txt")
#groundTruthList=readTxtIntoList(groundTruthDir+"/FileList.txt")
predictList = readTxtIntoList(PredictDir + "/FileList.txt")

#rawImageOutput=resultDir+"/Raw"
#groundTruthOutput=resultDir+"/Truth"
predictOutput = resultDir + "/Predict"

samplefactor=[2,2,2]
InterP = 0 
#zoomRawImageList = imagezoom(rawImageList, rawImageOutput, samplefactor, InterP)
#zoomTruthList = imagezoom(groundTruthList, groundTruthOutput,samplefactor, InterP)
zoomPredictList = imagezoom(predictList, predictOutput, samplefactor, InterP)
WriteListtoFile(zoomPredictList, resultDir + "/FileList.txt")