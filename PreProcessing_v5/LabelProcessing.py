"""
# Preprocessing for label data

# version 3

(1) Image Adjustment
(2) choose a reference
(3) Registration 

time: 07.06.2017

@author: louis
"""
import numpy as np # Numpy for general purpose processingd
import SimpleITK as sitk
import time
import subprocess
import os
import sys
import string
sys.path.insert(0, '../')
import DataPreProcessing.Func_data_preprocessing as Dpp
#------------------------------------------------------------------------------

RunImageAdjust = False
RundownSample = False
RunRegistration = True

IFdownSample = True; IFNormalization = False
InterP = 0 # Bilinear interpolation would be order=1, nearest is order=0, and cubic is the default (order=3).
root_dir = '/media/louis/Volume/ProgramWorkResult/VisercialMAS_ANT'
root_imge_dir = root_dir + '/Full_Image' + '/GC_Volumes_adjustment' # image root dir 
root_label_dir = root_dir + '/Full_Image' + '/GC_label_adjustment' # label root dir
reference_dir = root_imge_dir + '/referenceImage'
if IFNormalization != False: 
    reference_im_fn = reference_dir + '/reference_WithNorm' + '.nrrd'
else:
    reference_im_fn = reference_dir + '/reference_WithoutNorm' + '.nrrd'
#------------------------------------------------------------------------------
Regmode = 'linear'
#Regmode = 'nonlinear'

#organ = '40358_muscle_body_of_left_rectus_abdominis' 
#organ = '187_gallbladder'
#organ = '170_pancreas' 
#organ = '30325_left_adrenal_gland' 
#organ = '30324_right_adrenal_gland'
organ = '29193_first_lumbar_vertebra' 
#'''
# %%
# adjust the original Label data.                    
data_folder = '/media/louis/Volume/ResearchData/visceral_used/GC_Labels/'+ organ 
im_fns = Dpp.readTxtIntoList(data_folder +'/FileList.txt')

searchName = "10000129_1_CTce_ThAb"
for dir_i in im_fns:
    if dir_i.find(searchName) != -1:
        im_fns.remove (dir_i)
        break            

label_dir = root_label_dir + '/'+ organ
sizeadjust_label_dir = label_dir + '/sizeadjustment'
if not os.path.exists(sizeadjust_label_dir):
    subprocess.call('mkdir ' + '-p ' + sizeadjust_label_dir, shell=True) 
    
cuttingresult_dir = label_dir + '/cuttingresult'
if not os.path.exists(cuttingresult_dir):
    subprocess.call('mkdir ' + '-p ' + cuttingresult_dir, shell=True)  

#------------------------------------------------------------------------------
# counting the origin
    
ImageInfo = np.load('ImageInfo.npy')
ImInfo = ImageInfo.item()
OriginSet = ImInfo['origin']
DirectiongSet = ImInfo['direction']
SpacingSet = ImInfo['spacing']   

if RunImageAdjust != False:
    #--------------------------------------------------------------------------
    # discard the unuseful part
    
    wb_Num = 0
    modility = 'wb'
    for i in xrange(len(im_fns)):
        if modility in im_fns[i]:
            wb_Num=wb_Num+1
    
    cutrange = [[0.3,0.8],[0.0,1.0],[0.0,1.0]]    
    sizeadjust_list = Dpp.imageSizeAdjust(im_fns[0:wb_Num], sizeadjust_label_dir,\
                                          cutrange, SpacingSet, OriginSet, DirectiongSet)
    
    cutrange = [[0.0,1.0],[0.0,1.0],[0.0,1.0]]
    sizeadjust_list += Dpp.imageSizeAdjust(im_fns[wb_Num:], sizeadjust_label_dir,\
                                           cutrange, SpacingSet, OriginSet, DirectiongSet)
    Dpp.WriteListtoFile(sizeadjust_list, sizeadjust_label_dir+"/FileList.txt")

    #--------------------------------------------------------------------------
    # making the original images to same size
    cuttinginput_list = sizeadjust_list
    cuttingresult_list = Dpp.ImageCuttingwithDim(cuttinginput_list, cuttingresult_dir, ImInfo['dimCut'])    
    Dpp.WriteListtoFile(cuttingresult_list, cuttingresult_dir+"/FileList.txt")

#%%
if IFdownSample != False:

    downsampleinput_dir = cuttingresult_dir+ '/FileList.txt'
    downsampleinput_list = Dpp.readTxtIntoList(downsampleinput_dir)
    downsampleresult_dir = label_dir + '/downsampleresult'
    if not os.path.exists(downsampleresult_dir):
        subprocess.call('mkdir ' + '-p ' + downsampleresult_dir, shell=True)      
    
    if RundownSample != False:
        #--------------------------------------------------------------------------
        # downsample
        # Bilinear interpolation would be order=1, nearest is order=0, and cubic is the default (order=3).
        InterP = 0 
        downsamplefactor = (0.5, 0.5, 0.5) # for different dimension
        downsampleresult_list = Dpp.imageDownSample(downsampleinput_list, downsampleresult_dir,
                                                    downsamplefactor, InterP, SpacingSet, OriginSet, DirectiongSet)
        Dpp.WriteListtoFile(downsampleresult_list, downsampleresult_dir + '/FileList.txt')
#%%
# Registration

if IFdownSample != False:
    RegInput_dir = downsampleresult_dir + '/FileList.txt'    
else:
    RegInput_dir = cuttingresult_dir + '/FileList.txt'

RegInput_list = Dpp.readTxtIntoList(RegInput_dir)

if Regmode == 'linear':
    RegOutput_dir = label_dir + '/Regtrans_Linear_ANT'
    ImageRegOutput_dir = root_imge_dir + '/Regtrans_Linear_ANT'
elif Regmode == 'nonlinear':
    RegOutput_dir = label_dir + '/Regtrans_nonlinear_ANT'
    ImageRegOutput_dir = root_imge_dir + '/Regtrans_nonlinear_ANT'
else:
    raise ValueError('Regmode: linear/nonlinear')
if not os.path.exists(RegOutput_dir):
    subprocess.call('mkdir ' + '-p ' + RegOutput_dir, shell=True)
#------------------------------------------------------------------------------
    
if RunRegistration != False: 
    s = time.time()
    
    logFile = open(RegOutput_dir+'/RegTrans' + '_RUN_' + '.log', 'w')  

    fixedIm =  reference_im_fn
    
    transformParaMaplist = []    
    for i in RegInput_list:
        name, ext = os.path.splitext(i)
        labelBaseName = os.path.basename(name)
        imageBaseName = string.join(labelBaseName.split("_")[0:-2], "_")
        transformParaMaplist.append(ImageRegOutput_dir+'/ImRegResult_' + imageBaseName) 

    LabImOutputList = []
    ps = [] # to use multiple processors
    
    for i in range(len(RegInput_list)):              
        
        movingIm = RegInput_list[i]
        # write the result image to the file 
        name, ext = os.path.splitext(movingIm)
        labelBaseName = os.path.basename(name)
        LabImOutput = RegOutput_dir + '/LabelRegResult_' + labelBaseName + '.nrrd'
        LabImOutputList.append(LabImOutput)
                
        transformParadir_Trans = transformParaMaplist[i]

        cmd = ''
        cmd += Dpp.ANTSWarpImage(movingIm, LabImOutput, reference_im_fn, transformParadir_Trans, Regmode)
        process = subprocess.Popen(cmd, stdout = logFile, shell = True)
        ps.append(process)

    for p in ps:
        p.wait()

    Dpp.WriteListtoFile(LabImOutputList, RegOutput_dir+"/FileList.txt")    
    e = time.time()
    l = e - s
    
    print 'Registration is finished'
    print 'Total running time: %f mins'%(l/60.0)