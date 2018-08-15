# 
'''
# version 1
# louis
# 02.07.2017
# modify the images after affineTransform to binary image
'''

import sys
import os
import subprocess
sys.path.insert(0, '../')
import DataPreProcessing.Func_data_preprocessing as Dpp

regIfLinear = "Linear"
#regIfLinear = "nonlinear"

organ = '187_gallbladder'
#organ = '170_pancreas' 
#organ = '30325_left_adrenal_gland' 
#organ = '30324_right_adrenal_gland'
#organ = '29193_first_lumbar_vertebra' 

root_dir = '/media/louis/Volume/ProgramWorkResult/VisercialMAS_ANT'

label_data_dir = root_dir + '/Full_Image/GC_label_adjustment' + '/'+organ +\
                 '/Regtrans_' + regIfLinear +'_ANT'
result_dir_lab = root_dir + '/Full_Image/GC_label_adjustment' + '/'+organ +\
                 '/Modification_Regtrans_' + regIfLinear + '_ANT' 

print 'Results will be stored in:', result_dir_lab
if not os.path.exists(result_dir_lab):
    subprocess.call('mkdir ' + '-p ' + result_dir_lab, shell=True)

#--------------------------------------------------------------------------------
im_fns_label = Dpp.readTxtIntoList(label_data_dir + '/FileList.txt')
result_lab_list = Dpp.ImageModificationAfterTransform(im_fns_label, result_dir_lab)
Dpp.WriteListtoFile(result_lab_list, result_dir_lab + '/FileList.txt')
