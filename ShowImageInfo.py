# show image information

import SimpleITK as sitk # SimpleITK to load images
import sys
sys.path.insert(0, '../')
import DataPreProcessing.Func_data_preprocessing as Dpp

#%%
data_folder = '/media/louis/Volume/ProgramWorkResult/MICCAI_2017_6_12/Full_Image/GC_Volumes_adjustment/cuttingresult'
InputImageList = Dpp.readTxtIntoList(data_folder +'/FileList.txt')

for i in xrange(len(InputImageList)):    
    image = sitk.ReadImage(InputImageList[i])
    image_array = sitk.GetArrayFromImage(image) # get numpy array
    print 'Image No. %d'%(i)
    print 'dim z,x,y :'+str(image_array.shape)
    print 'Origin:' + str(image.GetOrigin())        
    print 'sapcing:' + str(image.GetSpacing())


