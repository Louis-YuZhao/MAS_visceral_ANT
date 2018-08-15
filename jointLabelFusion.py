# -*- coding: utf-8 -*-
import os
import subprocess
import string

#%%
def MultiAtlasJointLabelFusion(atlasImageList, atlasSegmentationlist, TargetImageList, OutputDir, EXECUTE = False):
    """
    Joint Label Fusion: 
    usage: 
    jointfusion dim mod [options] output_image

    required options:
    dim                             Image dimension (2 or 3)
    mod                             Number of modalities or features
    -g atlas1_mod1.nii atlas1_mod2.nii ...atlasN_mod1.nii atlasN_mod2.nii ... 
                                    Warped atlas images
    -tg target_mod1.nii ... target_modN.nii 
                                    Target image(s)
    -l label1.nii ... labelN.nii    Warped atlas segmentation
    -m <method> [parameters]        Select voting method. Options: Joint (Joint Label Fusion) 
                                    May be followed by optional parameters in brackets, e.g., -m Joint[0.1,2].
                                    See below for parameters
    other options: 
    -rp radius                      Patch radius for similarity measures, scalar or vector (AxBxC) 
                                    Default: 2x2x2
    -rs radius                      Local search radius.
                                    Default: 3x3x3
    -x label image.nii              Specify an exclusion region for the given label.
    -p filenamePattern              Save the posterior maps (probability that each voxel belongs to each label) as images.
                                    The number of images saved equals the number of labels.
                                    The filename pattern must be in C printf format, e.g. posterior%04d.nii.gz
    Parameters for -m Joint option:
    alpha                           Regularization term added to matrix Mx for inverse
                                    Default: 0.1
    beta                            Exponent for mapping intensity difference to joint error
                                    Default: 2
    """

#%%
    dim = 3
    mod = 1
    executable = '/home/louis/Documents/Packages/PICSL_MALF/jointfusion'
    result_folder = os.path.dirname(OutputDir)

    atlasImageStr = ' '
    atlasImageStr = atlasImageStr.join(atlasImageList)
    
    atlasSegmentationStr = ' '
    atlasSegmentationStr = atlasSegmentationStr.join(atlasSegmentationlist)
    
    TargetImageStr = ' '
    TargetImageStr = TargetImageStr.join(TargetImageList)
    
    name, ext = os.path.splitext(TargetImageList[0])
    BaseName = os.path.basename(name)
    testIndicator = string.join(BaseName.split("_")[-4:], "_")
        

    arguments = ' ' + str(dim) \
              + ' ' + str(mod)\
              + ' -g ' + atlasImageStr \
              + ' -l ' + atlasSegmentationStr \
              + ' -tg ' + TargetImageStr \
              + ' -m ' + 'Joint[0.1,2]' \
              + ' -p ' + result_folder+'/'+testIndicator+'_posterior%04d.nii.gz'\
              + ' ' + OutputDir \

    cmd = executable + ' ' + arguments
    if (EXECUTE):
        tempFile = open(result_folder+'/jointfusion_run.log', 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()
    return cmd