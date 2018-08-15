import os
import subprocess
from jointLabelFusion import MultiAtlasJointLabelFusion

#%%
atlasImageList = ['/media/data/louis/ProgramWorkResult/JointLabelFusionTest/Image_10000005_1_CT_wb.nrrd','/media/data/louis/ProgramWorkResult/JointLabelFusionTest/Image_10000006_1_CT_wb.nrrd']
atlasSegmentationlist = ['/media/data/louis/ProgramWorkResult/JointLabelFusionTest/Label_10000005_1_CT_wb_170_7.nrrd', '/media/data/louis/ProgramWorkResult/JointLabelFusionTest/Label_10000006_1_CT_wb_170_8.nrrd']
TargetImageList = ['/media/data/louis/ProgramWorkResult/JointLabelFusionTest/Image_10000011_1_CT_wb.nrrd',
                   '/media/data/louis/ProgramWorkResult/JointLabelFusionTest/Image_10000014_1_CT_wb.nrrd']
OutputDir = '/media/data/louis/ProgramWorkResult/JointLabelFusionTest/test1.nrrd'
if not os.path.exists(OutputDir):
    subprocess.call('mkdir ' + '-p ' + OutputDir, shell=True)
outputfile = os.path.join(OutputDir,'test.nrrd')    
cmd = MultiAtlasJointLabelFusion(atlasImageList, atlasSegmentationlist, TargetImageList, OutputDir, EXECUTE = True)