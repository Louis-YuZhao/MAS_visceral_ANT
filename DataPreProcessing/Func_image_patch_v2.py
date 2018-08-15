"""
# Founction of image Patch

# version 1

time:30.01.2017

@author: louis
"""
#%%
import os
import numpy as np
import SimpleITK as sitk # SimpleITK to load images

#%%
def sitk_ImageRead(image_dir):
    im = sitk.ReadImage(image_dir)
    im_array = sitk.GetArrayFromImage(im)
    return im_array    
    
def image_patch_calculate(im_dir_list):
    '''
    # cauculate the range and the central point of nonzero part    
    '''
    dim_store = np.zeros((6,len(im_dir_list)))
    for i in range(len(im_dir_list)):
        im_array = sitk_ImageRead(im_dir_list[i])
        im_nonzeros = np.transpose(np.nonzero(im_array > 10**(-2)))
        im_nonzeros_sort = np.sort(im_nonzeros, axis=0)
        dim_store[0:3,i] = im_nonzeros_sort[1]
        dim_store[3:6,i] = im_nonzeros_sort[-1]
    return dim_store

def patch_indice (im_dir_list):
    '''
    # cauculate the the index of patch part    
    '''
    dim_stroe = image_patch_calculate(im_dir_list)
    dim_stroe_first = dim_stroe[0:3,:]
    dim_stroe_second = dim_stroe[3:6,:]
    dim_stroe_inf = np.amin(dim_stroe_first, axis = 1)
    dim_stroe_sup = np.amax(dim_stroe_second, axis = 1)
    central_point = (dim_stroe_inf + dim_stroe_sup)//2
    dim_range_sup = np.abs(dim_stroe_sup-dim_stroe_inf)//2+1
    return central_point,dim_range_sup

def image_patch_padding (im_dir_list, central_point, dim_range_sup, reference_image, outputPrefix):
    im_ref = sitk.ReadImage(reference_image)

    PatchIndice = np.zeros((6,))
    PatchIndice[0:3] = central_point - dim_range_sup
    PatchIndice[3:6] = central_point + dim_range_sup
    PatchIndice = PatchIndice.astype(int)
    centralPointFinal = np.zeros((3,len(im_dir_list)))   

    outputlist = []
    for i in range(len(im_dir_list)):
        im = sitk.ReadImage(im_dir_list[i])
        im_array = sitk.GetArrayFromImage(im)
        im_z, im_x, im_y = im_array.shape
        z_begin = PatchIndice[0]
        z_end = PatchIndice[3]
        x_begin = PatchIndice[1]
        x_end = PatchIndice[4]
        y_begin = PatchIndice[2]
        y_end = PatchIndice[5]
        
        if z_begin < 0:
            pad_z = np.abs(0 - z_begin)
            patch_im_array = im_array[(z_begin+pad_z):(z_end+pad_z), :, :]
            centralPointFinal[:,i] = central_point + np.array((pad_z,0,0))
        elif z_end > im_z:
            pad_z = np.abs(z_end - im_z)
            patch_im_array = im_array[(z_begin-pad_z):(z_end-pad_z), :, :]
            centralPointFinal[:,i] = central_point - np.array((pad_z,0,0))
        else:
            patch_im_array = im_array[z_begin:z_end, :, :]
            centralPointFinal[:,i] = central_point
        
        if x_begin < 0:
            pad_x = np.abs(0 - x_begin)
            patch_im_array = patch_im_array[:, (x_begin+pad_x):(x_end+pad_x), :]
            centralPointFinal[:,i] = central_point + np.array((0,pad_x,0))
        elif x_end > im_x:
            pad_x = np.abs(x_end - im_x)
            patch_im_array = patch_im_array[:, (x_begin-pad_x):(x_end-pad_x), :]
            centralPointFinal[:,i] = central_point - np.array((0,pad_x,0))
        else:
            patch_im_array = patch_im_array[:, x_begin:x_end, :]
            centralPointFinal[:,i] = central_point
                
        if y_begin < 0:
            pad_y = np.abs(0 - y_begin)
            patch_im_array = patch_im_array[:, :, (y_begin+pad_y):(y_end+pad_y)]
            centralPointFinal[:,i] = central_point + np.array((0,0,pad_y))
        elif y_end > im_y:
            pad_y = np.abs(y_end - im_y)
            patch_im_array = patch_im_array[:, :, (y_begin-pad_y):(y_end-pad_y)]
            centralPointFinal[:,i] = central_point - np.array((0,0,pad_y))
        else:
            patch_im_array = patch_im_array[:, :, y_begin:y_end]
            centralPointFinal[:,i] = central_point 
                
        img = sitk.GetImageFromArray(patch_im_array)
        img.SetOrigin(im_ref.GetOrigin())
        img.SetSpacing(im_ref.GetSpacing())
        img.SetDirection(im_ref.GetDirection())
        
        name, ext = os.path.splitext(im_dir_list[i])
        baseName = os.path.basename(name)
        fn = outputPrefix +'/patch_'+ baseName+ '.nrrd'
        outputlist.append(fn)
        sitk.WriteImage(img,fn)
        print "the %d th patch is finished" %(i)
     
    return outputlist, centralPointFinal

def make_image_mask (im_dir_list, central_point, dim_range_sup, reference_image, outputPrefix):
    im_ref = sitk.ReadImage(reference_image)
     
    PatchIndice = np.zeros((6,len(im_dir_list)))
    PatchIndice[0:3,:] = central_point - dim_range_sup
    PatchIndice[3:6,:] = central_point + dim_range_sup
    PatchIndice = PatchIndice.astype(int)

    outputlist = []
    for i in range(len(im_dir_list)):
        im = sitk.ReadImage(im_dir_list[i])
        im_array = sitk.GetArrayFromImage(im)
        im_z, im_x, im_y = im_array.shape
        maskArray = np.zeros(im_array.shape, dtype = np.uint16)
        
        z_begin = PatchIndice[0,i]
        z_end = PatchIndice[3,i]
        x_begin = PatchIndice[1,i]
        x_end = PatchIndice[4,i]
        y_begin = PatchIndice[2,i]
        y_end = PatchIndice[5,i]
        
        if z_begin < 0:
            pad_z = np.abs(0 - z_begin)
        elif z_end > im_z:
            pad_z = np.abs(im_z - z_end)
        else:
            pad_z = 0
        
        if x_begin < 0:
            pad_x = np.abs(0 - x_begin)
        elif x_end > im_x:
            pad_x = np.abs(im_x - x_end)
        else:
            pad_x = 0
                
        if y_begin < 0:
            pad_y = np.abs(0 - y_begin)
        elif y_end > im_y:
            pad_y = np.abs(im_y - y_end)
        else:
            pad_y = 0    
            
        maskArray[(z_begin+pad_z):(z_end+pad_z),(x_begin+pad_x):(x_end+pad_x),(y_begin+pad_y):(y_end+pad_y)] = np.uint16(1)
                
        img = sitk.GetImageFromArray(maskArray)
        img.SetOrigin(im_ref.GetOrigin())
        img.SetSpacing(im_ref.GetSpacing())
        img.SetDirection(im_ref.GetDirection())
        
        name, ext = os.path.splitext(im_dir_list[i])
        baseName = os.path.basename(name)
        fn = outputPrefix +'/Mask_'+ baseName+ '.nrrd'
        outputlist.append(fn)
        img = sitk.Cast(img, sitk.sitkUInt16)
        sitk.WriteImage(img,fn)
        print "the %d th patch is finished" %(i)
     
    return outputlist