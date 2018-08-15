"""
# Founction of image PreProcessing

# version 2

time:34.08.2017

@author: louis

# Using Elastix for Registration
"""

import numpy as np # Numpy for general purpose processing
import SimpleITK as sitk # SimpleITK to load images
import subprocess
import os
import scipy.ndimage
import collections as col
from sklearn.preprocessing import normalize

#%%
def readTxtIntoList(filename):
   flist = []
   with open(filename) as f:
         flist = f.read().splitlines()
   return flist

def GetListFromFile(path, endwith):
    # get documents' list from a file
    fileList = []
    for root, dirs, files in list(os.walk(path)):
        for i in files :
            if i.endswith(endwith):
                fileList.append(root + "/" + i)
    return fileList

def WriteListtoFile(filelist, filename):
    with open(filename, 'w') as f:
        for i in filelist:
            f.write(i+'\n')
    return 1

def WriteDocumstoFile(inputpath, endwith, outputpath):
    with open(outputpath, 'w') as f:
        for root, dirs, files in list(os.walk(inputpath)):
            for i in files :
                if i.endswith(endwith):
                    tempdir = root + "/" + i
                    f.write(tempdir+'\n')
    return 1

def ImageDirListtoMatrix(InputImageList):
    N = len(InputImageList)
    image = sitk.ReadImage(InputImageList[0])
    image_array = sitk.GetArrayFromImage(image)
    dim = image_array.shape
    ImPara = {}
    ImPara['dim'] = dim
    ImPara['origin'] = image.GetOrigin()
    ImPara['spacing'] = image.GetSpacing()
    ImPara['direction'] = image.GetDirection()
    z_dim = dim[0]; x_dim = dim[1]; y_dim = dim[2]
    OutputMatrix = np.zeros((z_dim*x_dim*y_dim, N))
    OutputMatrix[:,0] = np.reshape(image_array,-1)
    for i in range(1,N):
        image = sitk.ReadImage(InputImageList[i])
        image_array = sitk.GetArrayFromImage(image) # get numpy array
        OutputMatrix[:,i] = np.reshape(image_array,-1)

    return OutputMatrix.astype(np.float32), ImPara

def saveImagesFromDM(dataMatrix, inputList, outputdir, referenceImName):
    im_ref = sitk.ReadImage(referenceImName)
    im_ref_array = sitk.GetArrayFromImage(im_ref) # get numpy array
    z_dim, x_dim, y_dim = im_ref_array.shape # get 3D volume shape

    outputlist = []
    num_of_data = dataMatrix.shape[1]
    for i in range(num_of_data):
        im = np.array(dataMatrix[:,i]).reshape(z_dim,x_dim,y_dim)
        img = sitk.GetImageFromArray(im)
        img.SetOrigin(im_ref.GetOrigin())
        img.SetSpacing(im_ref.GetSpacing())
        img.SetDirection(im_ref.GetDirection())

        name, ext = os.path.splitext(inputList[i])
        baseName = os.path.basename(name)
        fn = outputdir + baseName + '.nrrd'
        outputlist.append(fn)
        sitk.WriteImage(img,fn)
    del im_ref,im_ref_array
    return outputlist

#%%
# image size adjusting
def imageSizeAdjust(InputImageList, outputdir, cutrange, SpacingSet=None, Originset = None, DirectiongSet = None):
    outputlist = []
    for i in range(len(InputImageList)):
        image = sitk.ReadImage(InputImageList[i])
        image_array = sitk.GetArrayFromImage(image) # get numpy array
        z_dim, x_dim, y_dim = image_array.shape

        # image cutting
        zBegin = int(cutrange[0][0]*z_dim)
        zEnd = int (cutrange[0][1]*z_dim)
        xBegin = int (cutrange[1][0]*x_dim)
        xEnd = int (cutrange[1][1]*x_dim)
        yBegin = int (cutrange[2][0]*y_dim)
        yEnd = int (cutrange[2][1]*y_dim)
        im_array_cutting = image_array[zBegin:zEnd, xBegin:xEnd, yBegin:yEnd]
        Im = sitk.GetImageFromArray(im_array_cutting)

        if Originset != None:
            Im.SetOrigin(Originset)
        else:
            Im.SetOrigin(image.GetOrigin())

        if SpacingSet != None:
            Im.SetSpacing(SpacingSet)
        else:
            Im.SetSpacing(image.GetSpacing())

        if DirectiongSet != None:
            Im.SetDirection(DirectiongSet)
        else:
            Im.SetDirection(image.GetDirection())

        name, ext = os.path.splitext(InputImageList[i])
        name1, ext1 = os.path.splitext(name)
        baseName = os.path.basename(name1)
        fn = outputdir + '/sizeAdjust_' + baseName + '.nrrd'
        outputlist.append(fn)
        sitk.WriteImage(Im,fn)
        print 'SizeAdjust Num %d is finished'% i
    return outputlist

def Origin_computing(InputImageList):
    N = len(InputImageList)
    Origins = np.zeros((3,N))
    for i in range(N):
        image = sitk.ReadImage(InputImageList[i])
        Origins[:,i] = image.GetOrigin()
    return tuple(np.mean(Origins, axis=1))

def ImageCutting(InputImageList, cuttingresult_dir):
    outputlist = []
    N = len(InputImageList)
    DimStore = np.zeros((3,N))
    for i in range(N):
        image = sitk.ReadImage(InputImageList[i])
        image_array = sitk.GetArrayFromImage(image) # get numpy array
        DimStore[:,i] = image_array.shape
    DimStore = DimStore.astype(np.int32)

    Dimminum = np.amin(DimStore, axis=1)
    Dimmin = Dimminum[:,np.newaxis]
    DimDiff = DimStore - Dimmin
    DimBegin = DimDiff//2

    for i in range(N):
        image = sitk.ReadImage(InputImageList[i])
        image_array = sitk.GetArrayFromImage(image) # get numpy array
        image_array_cut = image_array[DimBegin[0,i]:(DimBegin[0,i]+Dimmin[0,0]), DimBegin[1,i]:(DimBegin[1,i]+Dimmin[1,0]), DimBegin[2,i]:(DimBegin[2,i]+Dimmin[2,0])]

        img = sitk.GetImageFromArray(image_array_cut)
        img.SetOrigin(image.GetOrigin())
        img.SetSpacing(image.GetSpacing())
        img.SetDirection(image.GetDirection())

        name, ext = os.path.splitext(InputImageList[i])
        baseName = os.path.basename(name)
        fn = cuttingresult_dir + '/CutResult_' + baseName + '.nrrd'
        outputlist.append(fn)
        sitk.WriteImage(img,fn)
        print 'Cutting Num %d is finished'% i
    return (outputlist, Dimminum)

def ImageCuttingwithDim(InputImageList, cuttingresult_dir, dim):
    outputlist = []
    N = len(InputImageList)
    for i in range(N):
        image = sitk.ReadImage(InputImageList[i])
        image_array = sitk.GetArrayFromImage(image) # get numpy array
        z_dim,x_dim,y_dim = image_array.shape

        if z_dim >= dim[0] and x_dim >= dim[1] and y_dim >= dim[2]:

            z_begin = z_dim//2 - dim[0]//2
            z_end = z_begin + dim[0]
            x_begin = x_dim//2 - dim[1]//2
            x_end = x_begin + dim[1]
            y_begin = y_dim//2 - dim[2]//2
            y_end = y_begin + dim[2]

            image_array_cut = image_array[z_begin : z_end, x_begin : x_end, y_begin : y_end]

            img = sitk.GetImageFromArray(image_array_cut)
            img.SetOrigin(image.GetOrigin())
            img.SetSpacing(image.GetSpacing())
            img.SetDirection(image.GetDirection())

            name, ext = os.path.splitext(InputImageList[i])
            baseName = os.path.basename(name)
            fn = cuttingresult_dir + '/CutResult_' + baseName + '.nrrd'
            outputlist.append(fn)
            sitk.WriteImage(img,fn)
            print 'Cutting Num %d is finished'% i
        else:
            print 'the unappropriate image:'+ InputImageList[i]
            continue
    return outputlist

def imageDownSample(InputImageList, output, downsamplefactor, InterP, SpacingSet=None, Originset = None, DirectiongSet = None):
    outputlist = []
    num_data=len(InputImageList)
    for i in range(num_data):
        image = sitk.ReadImage(InputImageList[i])
        image_array = sitk.GetArrayFromImage(image) # get numpy array

        # image downsampering
        im = scipy.ndimage.interpolation.zoom(image_array, downsamplefactor, order = InterP)
        img = sitk.GetImageFromArray(im)

        if Originset != None:
            img.SetOrigin(Originset)
        else:
            img.SetOrigin(image.GetOrigin())

        if SpacingSet != None:
            img.SetSpacing(SpacingSet)
        else:
            img.SetSpacing(image.GetSpacing())

        if DirectiongSet != None:
            img.SetDirection(DirectiongSet)
        else:
            img.SetDirection(image.GetDirection())

        name, ext = os.path.splitext(InputImageList[i])
        baseName = os.path.basename(name)
        fn = output + '/Imdownsample_' + baseName + '.nrrd'
        outputlist.append(fn)
        sitk.WriteImage(img, fn)
        print 'DownSample Num %d is finished'% i

    return outputlist

#%%
# NonLocal means filtering
# Intensity Normalization
#sklearn.preprocessing.normalize

#%%
# Registration

def ANTsRegistration(fixedIm, movingIm, outputTransformPrefix,params,initialTransform = False, EXECUTE = False):
    executable = '/home/louis/Documents/Python/ANTs-build/bin/antsRegistration'

    arguments = ' --output [%s]' %(outputTransformPrefix)
              # ' --output [%s,%sWarped.nrrd]' %(outputTransformPrefix, outputTransformPrefix)
              # ' --output [%s,%sWarped.nrrd,%sInverseWarped.nrrd]' %(outputTransformPrefix, outputTransformPrefix, outputTransformPrefix)
              #+ ' --use-histogram-match'

    for i in range(len(params)):
        antsParams = params[i]
        for key, value in antsParams.items():
            if key == ' --metric ':
                value = antsParams[' --metric '] # "MI[%s,%s, 1,50]" %(fixedIm,movingIm)
                value = value.replace('fixedIm', fixedIm)
                value = value.replace('movingIm', movingIm)
            arguments += key + value

    if initialTransform != False:
        initialTransform = initialTransform.replace('fixedIm', fixedIm)
        initialTransform = initialTransform.replace('movingIm', movingIm)
        arguments += ' --initial-moving-transform  %s' %(initialTransform)

    cmd = executable + ' ' + arguments
    if (EXECUTE):
        tempFile = open(outputTransformPrefix+'ANTsReg_run.log', 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()
    return cmd


def ANTsRegParaInside(fixedIm, movingIm, outputTransformPrefix, mode, initialTransform = False, EXECUTE = False):

    commen = col.OrderedDict ([[' --dimensionality ', str(3)],\
                           [' -u ', '1'] ,\
                           [' -z ', '1']])

    antsParamsTranslation = col.OrderedDict ([[' --metric ', 'mattes[ fixedIm, movingIm, 1, 32, regular, 0.05 ]'], #metric \
                                              [' --transform ', 'translation[ 0.1 ]'],  #transformation type \
                                              [' --convergence ', '[1000,1.e-8,20]'],   #no. of iterations and stopping criteria \
                                              [' --smoothing-sigmas ', '4vox'],  #smoothing sigmas \
                                              [' -f ', '6'],  # scale factors 6= 1/6 original size \
                                              [' -l ', '1']])  # -l estimate learning

    antsParamsRigid = col.OrderedDict ([[' --metric ', 'mattes[ fixedIm, movingIm, 1, 32, regular, 0.1 ]'], #metric \
                                       [' --transform ', 'rigid[ 0.1 ]'],  #transformation type \
                                       [' --convergence ', '[1000x1000,1.e-8,20]'],  #no. of iterations and stopping criteria \
                                       [' --smoothing-sigmas ', '4x2vox'],  #smoothing sigmas \
                                       [' -f ', '4x2'], #scale factors  \
                                       [' -l ', '1']  # -l estimate learning rate
                                       ])

    antsParamsAffine = col.OrderedDict ([[' --metric ', 'mattes[ fixedIm, movingIm, 1, 32, regular, 0.1 ]'],  #metric \
                                        [' --transform ', 'affine[ 0.1 ]'],  #transformation type \
                                        [' --convergence ', '[10000x1111x5,1.e-8,20]'],  #no. of iterations and stopping criteria \
                                        [' --smoothing-sigmas ', '4x2x1vox'],  #smoothing sigmas \
                                        [' -f ', '3x2x1'],  #scale factors \
                                        [' -l ', '1']  # -l estimate learning rate
                                        ])

    antsParamsNonliner = col.OrderedDict ([[' --metric ', 'mattes[ fixedIm, movingIm, 1, 50, Regular, 0.90]'],  #metric \
                                          [' --transform ', 'SyN[0.1,1,0]'],  #transformation type \
                                          [' --convergence ', '[100x50x25,1e-6,10]'],  #no. of iterations and stopping criteria \
                                          [' --smoothing-sigmas ', '2x1x0vox'],  #smoothing sigmas \
                                          [' -f ', '4x2x1'],  # scale factors \
                                          [' -l ', '1'] # -l estimate learning rate
                                        ])

    if mode == 'linear':
        params=[commen, antsParamsTranslation, antsParamsRigid, antsParamsAffine]
    elif mode == 'nonlinear':
        params=[commen, antsParamsTranslation, antsParamsRigid, antsParamsAffine, antsParamsNonliner]
    else:
        print 'Mode should be linear or nonlinear'
        raise SystemExit    

    if (EXECUTE):
        cmd = ANTsRegistration(fixedIm, movingIm, outputTransformPrefix, params, initialTransform, True)
    else:
        cmd = ANTsRegistration(fixedIm, movingIm, outputTransformPrefix, params, initialTransform, False)
    return cmd

def ANTsTransImage(inputIm, outputIm, referenceIm, transformPrefix, mode, EXECUTE = False):
    dim = 3
    executable = '/home/louis/Documents/Python/ANTs-build/bin/antsApplyTransforms'
    result_folder = os.path.dirname(outputIm)

    arguments = ' --dimensionality ' + str(dim) \
              + ' --input ' + inputIm \
              + ' --output ' + outputIm \
              + ' -r ' + referenceIm \
              + ' --float ' + '1'

    if mode == 'linear':
        tl = transformPrefix+'0GenericAffine.mat'
        arguments += ' --transform ' + tl

    elif mode == 'nonlinear':
        tnonl = transformPrefix+'1Warp.nii.gz'
        tl = transformPrefix+'0GenericAffine.mat'
        arguments += ' --transform ' + tnonl
        arguments += ' --transform ' + tl
    else:
        print 'Mode should be linear or nonlinear'
        raise SystemExit

    cmd = executable + ' ' + arguments
    if (EXECUTE):
        tempFile = open(result_folder+'/ANTsTransImage_run.log', 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()
    return cmd


def ANTSWarpImage(inputIm, outputIm, referenceIm, transformPrefix, mode = 'linear', EXECUTE = False):
    dim = 3
    executable = '/home/louis/Documents/Python/ANTs-build/bin/WarpImageMultiTransform'
    result_folder = os.path.dirname(outputIm)

    if mode == 'linear':
        t = transformPrefix + '0GenericAffine.mat'

    elif mode == 'nonlinear':
        t = transformPrefix + '1Warp.nii.gz'
        t +=' ' + transformPrefix +'0GenericAffine.mat'

    else:
        print 'Mode should be linear or nonlinear'
        raise SystemExit


    arguments = str(dim) +' %s  %s  -R %s %s '%(inputIm, outputIm, referenceIm, t)
    cmd = executable + ' ' + arguments
    if (EXECUTE):
        tempFile = open(result_folder+'/ANTSWarpImage.log', 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()
    return cmd

#%%
def ImageModificationAfterTransform(InputImageList, result_dir, threshold = 10**(-2)):
    outputlist = []
    N = len(InputImageList)
    for i in range(N):
        image = sitk.ReadImage(InputImageList[i])
        image_array = sitk.GetArrayFromImage(image) # get numpy array

        image_array[image_array > threshold] = 1
        image_array[image_array <= threshold] = 0

        img = sitk.GetImageFromArray(image_array)
        img.SetOrigin(image.GetOrigin())
        img.SetSpacing(image.GetSpacing())
        img.SetDirection(image.GetDirection())

        name, ext = os.path.splitext(InputImageList[i])
        baseName = os.path.basename(name)
        fn = result_dir + '/afmodify_' + baseName + '.nrrd'
        outputlist.append(fn)
        sitk.WriteImage(img,fn)
    return outputlist

def ImageNormlization(InputImageList, result_dir):
    outputlist = []
    N = len(InputImageList)
    for i in range(N):
        image = sitk.ReadImage(InputImageList[i])
        image_array = sitk.GetArrayFromImage(image) # get numpy array

        shape = np.shape(image_array)
        vecArray = image_array.reshape(1,-1)
        vecArray = vecArray + np.abs(vecArray.min(axis=1))
        normArray = normalize(vecArray, norm='max')

        img = sitk.GetImageFromArray(normArray.reshape(shape))
        img.SetOrigin(image.GetOrigin())
        img.SetSpacing(image.GetSpacing())
        img.SetDirection(image.GetDirection())

        name, ext = os.path.splitext(InputImageList[i])
        baseName = os.path.basename(name)
        fn = result_dir + '/Normalized_' + baseName + '.nrrd'
        outputlist.append(fn)
        sitk.WriteImage(img,fn)
    return outputlist
