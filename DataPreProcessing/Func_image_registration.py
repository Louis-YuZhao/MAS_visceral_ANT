"""
# Founction of image registration
# version 2

time:21.05.2018

@author: louis

# Using ANT for Registration
"""

import numpy as np # Numpy for general purpose processing
import SimpleITK as sitk # SimpleITK to load images
import subprocess
import os
import scipy.ndimage
import collections as col
from sklearn.preprocessing import normalize

#%%
# Registration

def ANTsRegistration(fixedIm, movingIm, fixedMask, movingMask, outputTransformPrefix,params,initialTransform = False, EXECUTE = False):
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


def ANTsRegParaInside(fixedIm, movingIm, fixedMask, movingMask, outputTransformPrefix, mode, initialTransform = False, EXECUTE = False):

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
        cmd = ANTsRegistration(fixedIm, movingIm, fixedMask, movingMask, outputTransformPrefix, params, initialTransform, True)
    else:
        cmd = ANTsRegistration(fixedIm, movingIm, fixedMask, movingMask, outputTransformPrefix, params, initialTransform, False)
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