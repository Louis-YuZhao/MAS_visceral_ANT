# -*- coding: utf-8 -*-
'''
version 1.1
Louis
06.06.2017

Additional founctions for Eigenface_Knn for under-organ-size patch
(2) achievement live-one-out fashion

'''
#%%   
import sys
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, '../')
import EigenfaceKNN.Func_Eigenface_KNN as EigKNN
import EigenfaceKNN.Func_k_nearest_neighbor_sklearning as KNNSklearn

#%%
def Imagelist_to_4DimMatrix(im_file_list, pad_para):
    
    pad_width = pad_para['pad_width']
    mode = pad_para['mode']
    ConV = pad_para['constant_values'] 
    
    num_of_data = len(im_file_list)
    im_file =  im_file_list[0]
    inIm = sitk.ReadImage(im_file)
    tmp = sitk.GetArrayFromImage(inIm)
    z_dim, x_dim, y_dim = tmp.shape
    
    z_dim += pad_width[0][0] + pad_width[0][1]
    x_dim += pad_width[1][0] + pad_width[1][1]
    y_dim += pad_width[2][0] + pad_width[2][1]
    
    Y = np.zeros((z_dim, x_dim, y_dim, num_of_data))      
    
    tmp_pad = np.lib.pad(tmp, pad_width, mode, constant_values = ConV)
    Y[:,:,:,0] = tmp_pad
    del tmp, tmp_pad    
   
    for i in range(1,num_of_data) :
          im_file =  im_file_list[i]
          inIm = sitk.ReadImage(im_file)
          tmp = sitk.GetArrayFromImage(inIm)
          tmp_pad = np.lib.pad(tmp, pad_width, mode, constant_values = ConV)
          Y[:,:,:,i] = tmp_pad
          del tmp, tmp_pad
    return Y

def patch_Eigenface_data(Eigenface_basis_matrix, EigenBasisNum, train_image_matrix, test_image_matrix, RPCA_Para):
    
    lamda = RPCA_Para['lamda']
    tol = RPCA_Para['tol']
    eigfaces, image_mean, low_rank, sparse = EigKNN.Eigenface_basis_RPCA(Eigenface_basis_matrix, EigenBasisNum, lamda, tol)
    
    # X_train    
    x_train = EigKNN.Eigenface_feature(train_image_matrix, image_mean, eigfaces)  
    
    # X_test
    x_test = EigKNN.Eigenface_feature(test_image_matrix, image_mean, eigfaces)    
    
    return x_train, x_test

def patch_KNN (x_train, y_train, x_test, K = 1):
    '''
    x_train : training data  (N*D)
    y_train : training label
    x_test : test data (N*D)
    '''
    best_k = K
    
    # useing traditional methods
#    classifier = KNN.KNearestNeighbor()
#    classifier.train(x_train, y_train)
#    y_test_pred, selection= classifier.predict(x_test, k=best_k)
    
#   using Sklearn
    classifier = KNNSklearn.KNearestNeighbor(algorithm='auto',metric='minkowski', p=2)
    classifier.train(x_train, y_train)
    y_test_pred,selection = classifier.predict(x_test, best_k)  
    
    return y_test_pred, selection
    
# save 3D images from data matrix
def saveImagesFromDataMatrix(dataMatrix,outputlist,ImPara):
    
    Origin = ImPara['Origin']
    Spacing = ImPara['Spacing']
    Direction = ImPara['Direction']
    Dim = ImPara['Dim']
    z_dim = Dim[0]
    x_dim = Dim[1]
    y_dim = Dim[2]
    
    num_of_data = dataMatrix.shape[1]
    for i in range(num_of_data):
        im = np.array(dataMatrix[:,i]).reshape(z_dim,x_dim,y_dim)
        img = sitk.GetImageFromArray(im)
        img.SetOrigin(Origin)
        img.SetSpacing(Spacing)
        img.SetDirection(Direction)
        fn = outputlist[i]
        sitk.WriteImage(img,fn)   
    return

def PatchMove(ImageMatrix, MovePara):
    # input :
    # ImageMatrix Z*X*Y*N 
    # MovePare (zmove, xmove, ymove) should small than the PatchSize
    # output:
    # MovedImageMatrix
    z_move = MovePara['z_move']
    x_move = MovePara['x_move']
    y_move = MovePara['y_move']
    z_dim, x_dim, y_dim, N = np.shape(ImageMatrix)
    MovedImageMatrix = np.zeros_like(ImageMatrix)
    tempmatrix1 = np.zeros((z_dim, x_dim, y_dim))
    tempmatrix2 = np.zeros((z_dim, x_dim, y_dim))
    tempmatrix3 = np.zeros((z_dim, x_dim, y_dim))
    tempmatrix4 = np.zeros((z_dim, x_dim, y_dim))
    
    for i in xrange(N):
        tempmatrix1 = ImageMatrix[:,:,:,i]
        tempmatrix2[0:(z_dim-z_move),:,:] = tempmatrix1[z_move:z_dim,:,:]
        tempmatrix2[(z_dim-z_move):z_dim,:,:] = tempmatrix1[0:z_move,:,:]
        tempmatrix3[:,0:(x_dim-x_move),:] = tempmatrix2[:,x_move:x_dim,:]
        tempmatrix3[:,(x_dim-x_move):x_dim,:] = tempmatrix2[:,0:x_move,:]
        tempmatrix4[:,:,0:(y_dim-y_move)] = tempmatrix3[:,:,y_move:y_dim]
        tempmatrix4[:,:,(y_dim-y_move):y_dim] = tempmatrix3[:,:,0:y_move]
        MovedImageMatrix[:,:,:,i] = tempmatrix4
    return MovedImageMatrix

def PatchMoveInverse(MovedImageMatrix, MovePara):
    # input :
    # MovedImageMatrix Z*X*Y*N 
    # MovePare (zmove, xmove, ymove) should small than the PatchSize
    # output:
    # ImageMatrix
    z_move = MovePara['z_move']
    x_move = MovePara['x_move']
    y_move = MovePara['y_move']
    z_dim, x_dim, y_dim, N = np.shape(MovedImageMatrix)
    ImageMatrix = np.zeros_like(MovedImageMatrix)
    tempmatrix1 = np.zeros((z_dim, x_dim, y_dim))
    tempmatrix2 = np.zeros((z_dim, x_dim, y_dim))
    tempmatrix3 = np.zeros((z_dim, x_dim, y_dim))
    tempmatrix4 = np.zeros((z_dim, x_dim, y_dim))
    
    for i in xrange(N):
        tempmatrix4 = MovedImageMatrix[:,:,:,i]
        tempmatrix3[:,:,0:y_move] = tempmatrix4[:,:,(y_dim-y_move):y_dim]
        tempmatrix3[:,:,y_move:y_dim] = tempmatrix4[:,:,0:(y_dim-y_move)]
        tempmatrix2[:,0:x_move,:] = tempmatrix3[:,(x_dim-x_move):x_dim,:]
        tempmatrix2[:,x_move:x_dim,:] = tempmatrix3[:,0:(x_dim-x_move),:]
        tempmatrix1[0:z_move,:,:] = tempmatrix2[(z_dim-z_move):z_dim,:,:]
        tempmatrix1[z_move:z_dim,:,:] = tempmatrix2[0:(z_dim-z_move),:,:] 
        ImageMatrix[:,:,:,i] = tempmatrix1
    return ImageMatrix

def removePadding(paddedImageMatrix, paddim):
    # input :
    # paddedImageMatrix Z*X*Y*N 
    # paddim ((pad_z,pad_z),(pad_x,pad_x),(pad_y,pad_y))
    # output:
    # ImageMatrix
    
    z_pad_b=paddim[0][0]
    z_pad_e=paddim[0][1]
    x_pad_b=paddim[1][0]
    x_pad_e=paddim[1][1]
    y_pad_b=paddim[2][0]
    y_pad_e=paddim[2][1]
    z_dim, x_dim, y_dim, N = np.shape(paddedImageMatrix)
    z_dim_final = z_dim-z_pad_b-z_pad_e
    x_dim_final = x_dim-x_pad_b-x_pad_e
    y_dim_final = y_dim-y_pad_b-y_pad_e
    ImageMatrix = np.zeros((z_dim_final, x_dim_final, y_dim_final, N))
    for i in xrange(N):
        ImageMatrix[:,:,:,i] = paddedImageMatrix[z_pad_b:(z_pad_b+z_dim_final),x_pad_b:(x_pad_b+x_dim_final),y_pad_b:(y_pad_b+y_dim_final),i]
    return ImageMatrix    

#%%
def KNNPrediction_LocalPatch(InputDataDir, KNNPara, OriginalImPara, PatchSize, pad_para, RPCA_Para):
    
    p_z = PatchSize[0]
    p_x = PatchSize[1]
    p_y = PatchSize[2]
    vecLen = p_z * p_x * p_y
    dim = OriginalImPara['Dim'] 
    z_dim = dim[0]
    x_dim = dim[1]
    y_dim = dim[2]
    paddim = pad_para['pad_width']    
    z_dim_h = z_dim+paddim[0][0]+paddim[0][1]
    x_dim_h = x_dim+paddim[1][0]+paddim[1][1]
    y_dim_h = y_dim+paddim[2][0]+paddim[2][1]
    vecLentestpre = z_dim * x_dim * y_dim
    
    Num_z = z_dim_h//p_z
    Num_x = x_dim_h//p_x
    Num_y = y_dim_h//p_y     
    
    finalImPara = {}
    finalImPara['Origin'] = OriginalImPara['Origin']
    finalImPara['Spacing'] = OriginalImPara['Spacing']
    finalImPara['Direction'] = OriginalImPara['Direction'] 
    finalImPara['Dim'] = (z_dim_h, x_dim_h, y_dim_h)
    
    X_data_list = InputDataDir ['X_data_list']
    Y_data_list = InputDataDir ['Y_data_list']

    data_num = KNNPara['data_num']
    train_num = KNNPara['train_num']
    test_num = KNNPara['test_num']
    PcNo = KNNPara['PcNo']
    K = KNNPara['K']
    RandomChoice = KNNPara['RandomChoice']
#    MovePara = KNNPara['MovePara']
#    TrainImageforBasis = KNNPara['TrainImageforBasis']  

#------------------------------------------------------------------------------
    
    if RandomChoice == True:
        # random choice
        mask = np.random.choice(data_num, data_num, replace=False)
        mask = list(mask)
    else:
        mask = KNNPara['ImageMake']
    
    train_selected = mask[0 : train_num]
    test_selected = mask[train_num : train_num + test_num]
    
    # X_train_matrix
    im_file_list_train=[]
    for i in train_selected:
        im_file_list_train.append(X_data_list[i])
    train_image_matrix = Imagelist_to_4DimMatrix (im_file_list_train, pad_para)
#    train_image_matrix = PatchMove(train_image_matrix, MovePara)
    
    # Data for making eigenface basis
    im_file_list_Eigenface_basis = []
    im_file_list_SC = []
    #for i in TrainImageforBasis:
    #    im_file_list_SC.append(X_sc_data_dir+ '/ImRegResult_'+str(i)+ '.nrrd')
    im_file_list_Eigenface_basis = im_file_list_train + im_file_list_SC
    Eigenface_basis_matrix = Imagelist_to_4DimMatrix (im_file_list_Eigenface_basis, pad_para)
#    Eigenface_basis_matrix = PatchMove(Eigenface_basis_matrix, MovePara)
    
    # Y_train = Y_train_matrix
    im_file_list_train=[]
    for i in train_selected:
        im_file_list_train.append(Y_data_list[i])
    y_train = Imagelist_to_4DimMatrix (im_file_list_train, pad_para)
#    y_train = PatchMove(y_train, MovePara)
    
    # X_test_matrix
    im_file_list_test=[]
    for i in test_selected:
        im_file_list_test.append(X_data_list[i])
    test_image_matrix = Imagelist_to_4DimMatrix (im_file_list_test, pad_para)
#    test_image_matrix = PatchMove(test_image_matrix, MovePara)
    
    # Y_test = Y_test_matrix
    im_file_list_test=[]
    for i in test_selected:
        im_file_list_test.append(Y_data_list[i])
    y_test_matrix = Imagelist_to_4DimMatrix (im_file_list_test, pad_para)
    
    
    #%%
    TiN = test_image_matrix.shape[3]
    y_test_pred = np.zeros((z_dim_h, x_dim_h, y_dim_h, TiN))
    
    i = 0   
    for z_dim in range(0, p_z * Num_z, p_z):
        for x_dim in range(0, p_x * Num_x, p_x):
            for y_dim in range(0, p_y * Num_y, p_y):
                
                temp1 = Eigenface_basis_matrix[z_dim:(z_dim+p_z), x_dim:(x_dim+p_x), y_dim:(y_dim+p_y), :]
                temp2 = train_image_matrix[z_dim:(z_dim+p_z), x_dim:(x_dim+p_x), y_dim:(y_dim+p_y), :]
                temp3 = test_image_matrix[z_dim:(z_dim+p_z), x_dim:(x_dim+p_x), y_dim:(y_dim+p_y), :]
                temp4 = y_train[z_dim:(z_dim+p_z), x_dim:(x_dim+p_x), y_dim:(y_dim+p_y), :]
                               
                EBM = np.reshape (temp1,(vecLen,-1))
                TrainM = np.reshape (temp2,(vecLen,-1))
                TestM = np.reshape (temp3,(vecLen,-1))
                Ytrain = np.reshape (temp4,(vecLen,-1))
    
                del temp1, temp2, temp3, temp4
                
                Xtrain, Xtest = patch_Eigenface_data(EBM, PcNo, TrainM, TestM, RPCA_Para)
                
                Xtrain = Xtrain.T # N*D
                Ytrain = Ytrain.T            
                Xtest = Xtest.T
                                           
                y_test_pred_patch, dist_dice = patch_KNN (Xtrain, Ytrain, Xtest, K)
                y_test_pred[z_dim:(z_dim+p_z), x_dim:(x_dim+p_x), y_dim:(y_dim+p_y), :] = y_test_pred_patch.reshape(p_z, p_x, p_y, TiN)
                
                i+=1
                if i%100==0:
                    print 'i = %d'%(i)
                
    #%%
#    y_test_pred = PatchMoveInverse(y_test_pred, MovePara)
    y_test_pred = removePadding(y_test_pred, paddim)
    y_test_matrix = removePadding(y_test_matrix, paddim)
    y_test = np.reshape (y_test_matrix,(vecLentestpre,-1))    
    y_test_pred_vectorize = np.reshape(y_test_pred,(vecLentestpre,TiN))   
    return y_test_pred_vectorize, y_test