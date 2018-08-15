# -*- coding: utf-8 -*-
'''
version 1.1
Louis
01.12.2016

founction for Eigenface_Knn

'''
import numpy as np
import SimpleITK as sitk # SimpleITK to load images
import matplotlib.pyplot as plt
from Func_low_rank_atlas_iter_PCA import AffineReg, rpca

def readTxtIntoList(filename):
    flist = []
    with open(filename) as f:
        flist = f.read().splitlines()
        return flist    

# Affine registering each input image to the reference(healthy atlas)  image
# changed in 27.11.2016
def affineRegistrationStep(im_fn_list, Result_FileName,reference_im_fn):
    num_of_data = len(im_fn_list)
    for i in range(num_of_data):
        outputIm = Result_FileName+ str(i)  + '.nrrd'
        AffineReg(reference_im_fn,im_fn_list[i],outputIm)
        print('AffingRegistrating image %d finished' % i)
    return

# save 3D images from data matrix
def saveImagesFromDM(dataMatrix, outputdir_list, referenceImName):
    im_ref = sitk.ReadImage(referenceImName)
    im_ref_array = sitk.GetArrayFromImage(im_ref) # get numpy array
    z_dim, x_dim, y_dim = im_ref_array.shape # get 3D volume shape
    
    num_of_data = dataMatrix.shape[1]
    for i in range(num_of_data):
        im = np.array(dataMatrix[:,i]).reshape(z_dim,x_dim,y_dim)
        img = sitk.GetImageFromArray(im)
        img.SetOrigin(im_ref.GetOrigin())
        img.SetSpacing(im_ref.GetSpacing())
        img.SetDirection(im_ref.GetDirection())
        fn = outputdir_list[i]
        sitk.WriteImage(img,fn)
    del im_ref,im_ref_array
    return
    
def showImage(image_fn, title):
    image = sitk.ReadImage(image_fn) # image in SITK format
    image_array = sitk.GetArrayFromImage(image) # get numpy array
    z_dim, x_dim, y_dim = image_array.shape # get 3D volume shape    

    # display reference image
    fig = plt.figure(figsize=(30,20))
    plt.subplot(131)
    implot = plt.imshow(np.flipud(image_array[z_dim/2,:,:]),plt.cm.gray)
    plt.subplot(132)
    implot = plt.imshow(np.flipud(image_array[:,x_dim/2,:]),plt.cm.gray)
    plt.subplot(133)
    implot = plt.imshow(np.flipud(image_array[:,:,y_dim/2]),plt.cm.gray)
    plt.axis('off')
    plt.title(title, fontsize = 32)         
    del image, image_array
    return
    
def normalized(a, axis= 0, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)
    
def Imagelist_to_Matrix(im_file_list):
   
   # prepare data matrix for low-rank decomposition
    num_of_data = len(im_file_list)
    im_file =  im_file_list[0]
    inIm = sitk.ReadImage(im_file)
    tmp = sitk.GetArrayFromImage(inIm)
    z_dim, x_dim, y_dim = tmp.shape
    vector_length = z_dim * x_dim * y_dim
    Y = np.zeros((vector_length,num_of_data))         
    Y[:,0] = tmp.reshape(-1)
    
    for i in range(1,num_of_data) :
          im_file =  im_file_list[i]
          inIm = sitk.ReadImage(im_file)
          tmp = sitk.GetArrayFromImage(inIm)         
          Y[:,i] = tmp.reshape(-1)
          del tmp
    return Y
    
def Eigenface_basis_RPCA(ImageMatrix,Principal_component_order, lamda, tol):    
    # 28/11/2016
    # louis
    # fouction: computing the Eigenfaces
    
    # input:
    # ImageMatrix: the images matrix (each column is a image)
    # Principal_component_order: the number of the principal_component
    
    # output:
    # eigfaces
    
    Y = ImageMatrix   
    # low-rank and sparse decomposition
   
#    sys.stdout.write('start Low-Rank/Sparse' + '\n')
    low_rank, sparse, n_iter, rank, sparsity, sum_sparse = rpca(Y, lamda, tol) 
       
#    sys.stdout.write('End Low-Rank/Sparse' + '\n')
    
    # Eigenface basis computing
    
    Gamma = low_rank
    image_mean = np.mean(Gamma, axis=1)
    image_mean = image_mean.reshape(-1,1)
    
    Phi = Gamma - image_mean
    L = np.dot(Phi.T,Phi) # L = A.T A
    # Compute the eigenvalues and right eigenvectors of a square array.(The normalized (unit "length") eigenvectors)
    eigVals,eigVects=np.linalg.eig(L) 
    eigVects_U = np.dot(Phi, eigVects)
    eigVects_U = normalized(eigVects_U)
    eigfaces = eigVects_U[:,0:Principal_component_order]
    
    return eigfaces, image_mean, low_rank, sparse
    
    
def Eigenface_basis(ImageMatrix, Principal_component_order):
    # 16/11/2016
    # louis
    # fouction: computing the Eigenfaces
    
    # input:
    # ImageMatrix: the images matrix (each column is a image)
    # Principal_component_order: the number of the principal_component 
        
    # output:
    # eigfaces

    Y = ImageMatrix  
        
    # Eigenface basis computing
    
    Gamma = Y
    image_mean = np.mean(Gamma, axis=1)
    image_mean = image_mean.reshape(-1,1)
    Phi = Gamma - image_mean
    L = np.dot(Phi.T,Phi) # L = A.T A
    # Compute the eigenvalues and right eigenvectors of a square array.(The normalized (unit "length") eigenvectors)
    eigVals,eigVects=np.linalg.eig(L) 
    eigVects_U = np.dot(Phi, eigVects)
    eigVects_U = normalized(eigVects_U)
    eigfaces = eigVects_U[:,0:Principal_component_order]
    
    return eigfaces, image_mean
    
def Eigenface_feature(Matrix_image, image_mean, eigfaces):
    
    # 16/11/2016
    # louis
    # fouction: computing the eigenface components
    
    # input:
    # Matrix_image: image data (each column should represent the vectorized image D*N)
    # eigfaces: canculated eigfaces
        
    # output:
    # eigcom: eigenface components (each column represents the eigenface components)
    
    Phi = Matrix_image - image_mean
    eigcom = np.dot(eigfaces.T, Phi)
    
    return eigcom # PcNo*N

def DiceScoreCalculation(A,B):
    # 05/12/2016
    # louis
    # version 1.1
    # fouction: computing the dicescore
    
    k = 1
    constraint = 10**(-2)
    Nonzero_A = np.transpose(np.nonzero(np.abs(A) > constraint))
    Nonzero_B = np.transpose(np.nonzero(np.abs(B) > constraint))
    A[Nonzero_A]=k
    B[Nonzero_B]=k
    dice = np.sum(A[B==k])*2.0 / (np.sum(A) + np.sum(B))
    return dice  

#def
    

    

        
        
    