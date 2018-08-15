# founctions for Gaussian Weighted Majority Voting

'''
version: 1
time: 2017.02.14
authour: louis

'''
import numpy as np # Numpy for general purpose processing

#%%
def NormalizedL2NormDistance(PatchIn, PatchRef):
    # computting the L2NormDistance of PatchIn and PatchRef 
   
    z_dim, x_dim, y_dim = PatchIn.shape
    Dis = np.linalg.norm(PatchIn - PatchRef)
    # Dis = np.sqrt(np.sum(np.square(x)))
    NormalizedDis = Dis/(z_dim*x_dim*y_dim)                   
                  
    return NormalizedDis

def InVolumeandRefVolumePad(VolumIn, VolumRef, MajorityVotingPara):
    
    '''
    SN = MajorityVotingPara['SearchNeighborhood']
    PS = MajorityVotingPara['PatchSize']
    
    PSz, PSx, PSy = PS
    InPadNoz = PSz/2;
    InPadNox = PSx/2;
    InPadNoy = PSy/2;
    
    SNz, SNx, SNy = SN
    RefPadNoz = SNz/2 + InPadNoz;
    RefPadNox = SNx/2 + InPadNox;
    RefPadNoy = SNy/2 + InPadNoy;
    '''
    InPadNoz, InPadNox, InPadNoy = MajorityVotingPara['InPadNo']
    RefPadNoz, RefPadNox, RefPadNoy = MajorityVotingPara['RefPadNo']
    
    OutputVolumIn = np.pad(VolumIn, ((InPadNoz,InPadNoz), (InPadNox, InPadNox), (InPadNoy, InPadNoy)), 'constant') 
    OutputVolumRef = np.pad(VolumRef, ((RefPadNoz, RefPadNoz), (RefPadNox, RefPadNox), (RefPadNoy, RefPadNoy)), 'constant')
    
    return OutputVolumIn, OutputVolumRef

def MVWeightCouputing3D(VolumIn, VolumRef,VolumeY, MajorityVotingPara):
    # computting the Weight between VolumIn and VolumRef
    # VolumIn: Input Volum after padding
    # VolumRef: Ref Volum after padding
    '''   
    SN = MajorityVotingPara['SearchNeighborhood']
    PS = MajorityVotingPara['PatchSize']
    
    PSz, PSx, PSy = PS
    InPadNoz = PSz/2
    InPadNox = PSx/2
    InPadNoy = PSy/2
        
    SNz, SNx, SNy = SN    
    
    z_dim, x_dim, y_dim = VolumIn.shape    
    index1 = z_dim * x_dim * y_origen_dim
    index2 = SNz*SNx*SNy
    
    '''
    dim_z, dim_x, dim_y = MajorityVotingPara['OriginDim']
    SNz, SNx, SNy = MajorityVotingPara['SearchNeighborhood']
    indexD,indexNH = MajorityVotingPara['index']
    InPadNoz, InPadNox, InPadNoy = MajorityVotingPara['InPadNo']
    NBNoz, NBNox, NBNoy = MajorityVotingPara['NBNo']
    
    WL2N = np.zeros((indexD, indexNH))
    Y_sj = np.zeros((indexD, indexNH))
    
    VolumInpad, VolumRefpad = InVolumeandRefVolumePad(VolumIn, VolumRef, MajorityVotingPara)
    VolumeYpad = np.pad(VolumeY,((NBNoz, NBNoz),(NBNox, NBNox),(NBNoy, NBNoy)),'constant')
    
    for i in xrange(InPadNoz, dim_z+InPadNoz): # true i-InPadNoz
        for j in xrange(InPadNox, dim_x+InPadNox): # true j-InPadNox
            for k in xrange(InPadNoy, dim_y+InPadNoy):# true k-InPadNoy
                PatchIn = VolumInpad[(i-InPadNoz):(i+InPadNoz+1), (j-InPadNox):(j+InPadNox+1), (k-InPadNoy):(k+InPadNoy+1)]
                indexdim1 = (i-InPadNoz)*dim_x*dim_y + (j-InPadNox)*dim_y + (k-InPadNoy)
                
                for ii in xrange((-NBNoz),NBNoz+1):
                    for jj in xrange((-NBNox),NBNox+1):
                        for kk in xrange((-NBNoy),NBNoy+1):
                            PatchRef = VolumRefpad[(i-InPadNoz+ii+NBNoz):(i+InPadNoz+1+ii+NBNoz), (j-InPadNox+jj+NBNox):(j+InPadNox+1+jj+NBNox), (k-InPadNoy+kk+NBNoy):(k+InPadNoy+1+kk+NBNoy)]
                                                        
                            indexdim2 = (ii+NBNoz)*SNx*SNy + (jj+NBNox)*SNy + kk+NBNoy
                            WL2N[indexdim1,indexdim2] = NormalizedL2NormDistance(PatchIn, PatchRef)
                            Y_sj[indexdim1,indexdim2] = VolumeYpad[(i-InPadNoz)+(ii+NBNoz), (j-InPadNox)+(jj+NBNox), (k-InPadNoy)+(kk+NBNoy)]
#        print i
    
    return WL2N, Y_sj                            

def MVWeightCouputing4D(VolumeIn3D, VolumeRefs4D, VolumeY4D, MajorityVotingPara):
    '''
    # canculate the Weight and related Y
    # VolumeIn4D : (dim_z, dim_x, dim_y)
    # VolumeRefs4D : (K, dim_z, dim_x, dim_y)
    # VolumeY4D : (K, dim_z, dim_x, dim_y)
    
    '''
    N,K = MajorityVotingPara['NK']
    indexD,indexNH = MajorityVotingPara['index']
    W = np.zeros((indexD, indexNH, K))
    Y = np.zeros((indexD, indexNH, K))
    for k in xrange(K):
        W[:,:,k], Y[:,:,k] = MVWeightCouputing3D(VolumeIn3D, VolumeRefs4D[k,:,:,:],VolumeY4D[k,:,:,:], MajorityVotingPara)
      
    H = np.amin(np.amin(W,axis = 2), axis = 1)    
    H = H + 10**(-8)
    H = H[:,np.newaxis, np.newaxis]
    
    W = np.exp((-(W**2)/H))    
    return W, Y
    
def LabelFusion(W,Y):    
    temp1 = np.sum(np.sum(W*Y,axis = 2), axis = 1)
    temp2 = np.sum(np.sum(W,axis = 2), axis = 1)
    V = temp1/temp2
#    dim_z, dim_x, dim_y = DimPara
#    V_out = np.reshape(V,(dim_z, dim_x, dim_y))
    
    return V

def LabelResult(V, constraint):
    
    V[V >= constraint] = 1
    V[V < constraint] = 0
     
    return V

def WeightedMajorityVoting(x_test, x_selected, y_selected, OriginalImPara, MajorityVotingPara, WMVconstraint = 0.5):
    
    '''
    # x_test num_test*D
    # x_selected num_test*K*D
    # y_selected num_test*K*M
    # OriginalImPara: Image parameters
    
    '''
    N, K, D = x_selected.shape   

    Dim = OriginalImPara['Dim']    
    dim_z = Dim[0]
    dim_x = Dim[1]
    dim_y = Dim[2]
    
    vectorlen = D
    OutputLabels = np.zeros((vectorlen, N))
        
    SN = MajorityVotingPara['SearchNeighborhood']
    PS = MajorityVotingPara['PatchSize']
    
    PSz, PSx, PSy = PS
    InPadNoz = PSz/2
    InPadNox = PSx/2
    InPadNoy = PSy/2
    
    SNz, SNx, SNy = SN
    NBNoz = SNz/2
    NBNox = SNx/2
    NBNoy = SNy/2
        
    
    RefPadNoz = NBNoz + InPadNoz;
    RefPadNox = NBNox + InPadNox;
    RefPadNoy = NBNoy + InPadNoy;
           
    indexD = vectorlen
    indexNH = SNz*SNx*SNy
    
    MajorityVotingPara['OriginDim'] = (dim_z, dim_x, dim_y)
    MajorityVotingPara['InPadNo'] = (InPadNoz, InPadNox, InPadNoy)
    MajorityVotingPara['RefPadNo'] = (RefPadNoz, RefPadNox, RefPadNoy)
    MajorityVotingPara['NBNo'] = (NBNoz, NBNox, NBNoy)
    MajorityVotingPara['index'] = (indexD,indexNH)
    MajorityVotingPara['NK'] = (N,K)
    x_test_reshape = np.reshape(x_test, (N, dim_z, dim_x, dim_y))
    x_selected_reshape = np.reshape(x_selected, (N, K, dim_z, dim_x, dim_y))
    y_selected_reshape = np.reshape(y_selected, (N, K, dim_z, dim_x, dim_y))
    for i in xrange(N):
        VolumeIn = x_test_reshape[i, :, :, :]
        VolumeRefs = x_selected_reshape[i, :, :, :, :]
        VolumeY = y_selected_reshape[i, :, :, :, :]        
        Weight, Y = MVWeightCouputing4D(VolumeIn, VolumeRefs, VolumeY, MajorityVotingPara)
        V_out = LabelFusion(Weight,Y)
        label = LabelResult(V_out, WMVconstraint)
        OutputLabels[:,i] = label
        print i
    
    return OutputLabels

def MatrixSelect (XTrainMatrix, SelectNo): 
    
    '''
    # XTrainMatrix : N * D
    # SelectNo : num_test * K
    
    '''
    D = XTrainMatrix.shape[1]
    num_test, K = SelectNo.shape
    
    x_selected = np.zeros((num_test, K, D))
    
    dists_sort_dice=np.zeros((SelectNo.shape))
    dist_dice=np.zeros((num_test,K))      
    
    dists_sort_dice=np.argsort(SelectNo, axis=1)
    dist_dice =dists_sort_dice[:,0:K]
    for i in xrange(num_test):
      # A list of length K storing the labels of the K nearest neighbors to
      # the ith test point.     
      dice_i = np.array(dist_dice[i,:])
      dice_i = list(dice_i.reshape(-1))
      closest_x = XTrainMatrix[dice_i] # K*M      
      x_selected[i,:,:] = closest_x
                
    return x_selected