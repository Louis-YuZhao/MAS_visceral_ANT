"""
# Preprocessing for image volume data

# version 3

(1) downsample
(2) choose a reference
(3) Unitilize Elastix as the Registration Toolbox

time:08.24.2017

@author: louis
"""

import numpy as np # Numpy for general purpose processing
import SimpleITK as sitk # SimpleITK to load images
import subprocess
import os
import sys
#from skimage.exposure import equalize_hist
import time
sys.path.insert(0, '../')
from DataPreProcessing.Func_data_preprocessing import (readTxtIntoList,
WriteListtoFile,ImageDirListtoMatrix,saveImagesFromDM,Origin_computing, 
imageSizeAdjust, ImageCutting, imageDownSample, ANTsRegParaInside, ANTSWarpImage)
import DataPreProcessing.Func_atlas_selection as Aselec

#%%
#------------------------------------------------------------------------------
RunImageAdjust = False
RundownSample = False
RunMakeReference = True
RunNormalization = False
RunRegistration = False

IFdownSample = True; IFNormalization = False

InterP = 0 # Bilinear interpolation would be order=1, nearest is order=0, and cubic is the default (order=3).   
Regmode = 'nonlinear'

# output path
root_dir = '/media/data/louis/ProgramWorkResult/VisercialMAS_ANT'
root_image_dir = root_dir +'/Full_Image' + '/GC_Volumes_adjustment' # image root dir
root_label_dir = root_dir +'/Full_Image' + '/GC_label_adjustment' # image root dir

# Input path
data_folder = '/media/data/louis/ResearchData/visceral_used/GC_Volumes'
im_fns_raw = readTxtIntoList(data_folder +'/FileList.txt')
# adjust the GC original data.
Image_select = range(32)+range(33,40)
'''
there is no "10000129_1_CTce_ThAb.nii.gz"
'''
im_fns = []
for i in Image_select:
    im_fns.append(im_fns_raw[i])

#%%
sizeadjust_list=[]
sizeadjust_image_dir = root_image_dir + '/sizeadjustment'
if not os.path.exists(sizeadjust_image_dir):
    subprocess.call('mkdir ' + '-p ' + sizeadjust_image_dir, shell=True)

cuttingresult_dir = root_image_dir + '/cuttingresult'
if not os.path.exists(cuttingresult_dir):
    subprocess.call('mkdir ' + '-p ' + cuttingresult_dir, shell=True)

#------------------------------------------------------------------------------
# counting the origin
origin = Origin_computing(im_fns)
OriginSet = origin
DirectiongSet = (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
SpacingSet = (1, 1, 1)

ImInfo = {}
ImInfo['origin'] = OriginSet
ImInfo['direction'] = DirectiongSet
ImInfo['spacing'] = SpacingSet

if RunImageAdjust != False:
    #--------------------------------------------------------------------------
    # discard the unuseful part
    beginNum=0;
    WBNum=20
    endNum = len(im_fns)
    # WbCT
    cutrange = [[0.3,0.8],[0.0,1.0],[0.0,1.0]]
    sizeadjust_list = imageSizeAdjust(im_fns[beginNum:WBNum], sizeadjust_image_dir, cutrange, SpacingSet, OriginSet, DirectiongSet)
    # CeCT
    cutrange = [[0.0,1.0],[0.0,1.0],[0.0,1.0]]
    sizeadjust_list += imageSizeAdjust(im_fns[WBNum:endNum], sizeadjust_image_dir, cutrange, SpacingSet, OriginSet, DirectiongSet)
    WriteListtoFile(sizeadjust_list, sizeadjust_image_dir+"/FileList.txt")

    #----------------------------------cuttinginput_list-----------------------
    # making the original images to same size
    cuttinginput_list = sizeadjust_list
    cuttingresult_list, dim = ImageCutting(cuttinginput_list, cuttingresult_dir)
    WriteListtoFile(cuttingresult_list, cuttingresult_dir+"/FileList.txt")

    ImInfo['dimCut'] = (dim[0], dim[1], dim[2])
    np.save('ImageInfo', ImInfo)

#%%
if IFdownSample != False:
    #Read the relative directions from the file
    downsampleinput_dir = cuttingresult_dir+ '/FileList.txt'
    downsampleinput_list = readTxtIntoList(downsampleinput_dir)
    
    downsampleresult_dir = root_image_dir + '/downsampleresult'
    if not os.path.exists(downsampleresult_dir):
        subprocess.call('mkdir ' + '-p ' + downsampleresult_dir, shell=True)
    
    if RundownSample != False:
        #--------------------------------------------------------------------------
        # downsample
        # Bilinear interpolation would be order=1, nearest is order=0, and cubic is the default (order=3).        
        downsamplefactor = (0.5, 0.5, 0.5) # for different dimension
        downsampleresult_list = imageDownSample(downsampleinput_list, downsampleresult_dir, downsamplefactor, InterP, SpacingSet, OriginSet, DirectiongSet)
        WriteListtoFile(downsampleresult_list, downsampleresult_dir + '/FileList.txt')

#%%
# make the reference
if IFdownSample != False:
    makeReferenceInput_list = readTxtIntoList(downsampleresult_dir + '/FileList.txt')
else:
    makeReferenceInput_list = readTxtIntoList(cuttingresult_dir + '/FileList.txt')
    
reference_dir = root_image_dir +'/referenceImage'
if not os.path.exists(reference_dir):
    subprocess.call('mkdir ' + '-p ' + reference_dir, shell=True)

if RunMakeReference != False and IFNormalization == False:
    DataMatrix, ImPara = ImageDirListtoMatrix(makeReferenceInput_list) # DataMatrix =  # (z*x*y)*N

    SelImage, SelArg = Aselec.image_select_one(DataMatrix)

    image_array = np.reshape(SelImage,ImPara['dim'])
    img = sitk.GetImageFromArray(image_array)
    img.SetOrigin(ImPara['origin'])
    img.SetSpacing(ImPara['spacing'])
    img.SetDirection(ImPara['direction'])
    fn = reference_dir + '/reference_WithoutNorm'+'.nrrd'
    sitk.WriteImage(img,fn)
    
    name, ext = os.path.splitext(makeReferenceInput_list[SelArg])
    baseName = os.path.basename(name)
    fnRecord ='/reference_WithoutNorm_' + baseName + '.nrrd'
    WriteListtoFile([fnRecord], reference_dir+"/FileList.txt")

    ImageInfo = np.load('ImageInfo.npy')
    ImInfo = ImageInfo.item()
    ImInfo['dimAdj'] = ImPara['dim']
    np.save('ImageInfo', ImInfo)
    del SelImage, image_array
    
    print "the chose template is: " + makeReferenceInput_list[SelArg]

#%%
if IFNormalization != False:
    normalization_dir = root_image_dir + '/normalizationImage/NormalizedImage_'
    if not os.path.exists(normalization_dir):
        subprocess.call('mkdir ' + '-p ' + normalization_dir, shell=True)

    reference_dir = root_image_dir +'/referenceImage'
    if not os.path.exists(reference_dir):
        subprocess.call('mkdir ' + '-p ' + reference_dir, shell=True)
        
    if RunNormalization != False: 
        
        DataMatrix, ImPara = ImageDirListtoMatrix(makeReferenceInput_list) # DataMatrix =  # (z*x*y)*N
        from sklearn.preprocessing import normalize
        DataMatrix = DataMatrix + np.abs(DataMatrix.min(axis=0))
        NormalizedDataMatrix = normalize(DataMatrix, norm='max', axis = 0)
#        # histgramEquation        
#        HistMatrix = np.zeros(np.shape(NormalizedDataMatrix))
#        for i in xrange((np.shape(NormalizedDataMatrix))[1]):
#            HistMatrix[:,i] = equalize_hist(NormalizedDataMatrix[:,i])
        REF = makeReferenceInput_list[0] # just for getting the information of origin, direction and spacing.
        NormalizedOutput_list = saveImagesFromDM(NormalizedDataMatrix, makeReferenceInput_list, normalization_dir, REF)
        WriteListtoFile(NormalizedOutput_list, normalization_dir + '/FileList.txt')
    
        # making the reference
        InputImage = NormalizedDataMatrix  # (z*x*y)*N
        SelImage, SelArg = Aselec.image_select_one(InputImage)    

        image_array = np.reshape(SelImage,ImPara['dim'])
        img = sitk.GetImageFromArray(image_array)
        img.SetOrigin(ImPara['origin'])
        img.SetSpacing(ImPara['spacing'])
        img.SetDirection(ImPara['direction'])

        fn = reference_dir + '/reference_WithNorm' + '.nrrd'
        sitk.WriteImage(img,fn)
        
        name, ext = os.path.splitext(makeReferenceInput_list[SelArg])
        baseName = os.path.basename(name)
        fnRecord = '/reference_WithNorm_' + baseName + '.nrrd'
        WriteListtoFile([fnRecord], reference_dir+"/FileList.txt")
        
        ImageInfo = np.load('ImageInfo.npy')
        ImInfo = ImageInfo.item()
        ImInfo['dimAdj'] = ImPara['dim']
        np.save('ImageInfo', ImInfo)
        del SelImage, image_array
        
        print "the chose template is: " + makeReferenceInput_list[SelArg]
        
#%%
#  Registration

if IFNormalization != False:
    reference_im_fn = reference_dir + '/reference_WithNorm' + '.nrrd'
    RegInput_dir = normalization_dir + '/FileList.txt'
else:
    reference_im_fn = reference_dir + '/reference_WithoutNorm' + '.nrrd'
    RegInput_dir = downsampleresult_dir + '/FileList.txt'

RegInput_list = readTxtIntoList (RegInput_dir)
if Regmode == 'linear':
    RegOutput_dir = root_image_dir + '/Regtrans_Linear_ANT'
elif Regmode == 'nonlinear':
    RegOutput_dir = root_image_dir + '/Regtrans_nonlinear_ANT'
else: 
    raise ValueError('Regmode: linear/nonlinear')
if not os.path.exists(RegOutput_dir):
    subprocess.call('mkdir ' + '-p ' + RegOutput_dir, shell=True)

#------------------------------------------------------------------------------

if RunRegistration != False:
    s = time.time()
    MulProcNo = 6
    logFile = open(RegOutput_dir+'/RegTrans' + '_RUN_' + '.log', 'w')
   
    ps = [] # to use multiple processors    
    
    fixedIm =  reference_im_fn
    outputTransformPrefix = []  
    
    for i in range(len(RegInput_list)):    
        movingIm = RegInput_list[i]
        name, ext = os.path.splitext(movingIm)
        baseName = os.path.basename(name)
        transformPara = RegOutput_dir + '/ImRegResult_' + baseName
        outputTransformPrefix.append(transformPara) 
        
        cmd = ''
        # generate the warped input image with the specified file name
        initialTransform = '[fixedIm, movingIm, 1]'
        cmd += ANTsRegParaInside(fixedIm, movingIm, outputTransformPrefix[i], Regmode, initialTransform)
        process = subprocess.Popen(cmd, stdout = logFile, shell = True)
        ps.append(process)
        
        if (i+1) % MulProcNo == 0:
            for  p in ps:
                p.wait()
            ps = []
            print 'Processing Number: %d' %(i)
        
    for  p in ps:
        p.wait()    
    
    ps = [] # to use multiple processors
    RegImOutputList = []
    
    for i in range(len(RegInput_list)):                  
        inputIm = RegInput_list[i]
        name, ext = os.path.splitext(inputIm)
        baseName = os.path.basename(name)
        RegImOutput= RegOutput_dir+'/ImRegResult_' + baseName + '.nrrd'
        RegImOutputList.append(RegImOutput)
        transformPrefix = outputTransformPrefix[i]       
        
        cmd = ''
        cmd += ANTSWarpImage(inputIm, RegImOutput, reference_im_fn, transformPrefix, Regmode)   
        process = subprocess.Popen(cmd, stdout = logFile, shell = True)
        ps.append(process)    
        
        if (i+1) % MulProcNo == 0:
            for  p in ps:
                p.wait()
            ps = []
            print 'Processing Number: %d' %(i)
        
    for  p in ps:
        p.wait()
    
    WriteListtoFile(RegImOutputList, RegOutput_dir+"/FileList.txt")  
    e = time.time()
    l = e - s    
    print 'Registration is finished'
    print 'Total running time: %f mins'%(l/60.0)   