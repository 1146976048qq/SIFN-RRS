import numpy as np
import tensorflow.keras.utils
from tensorflow.keras import backend as K
import tensorflow as tf
# import scipy.io as sio


def generateTheta(L,endim):
    theta_=np.random.normal(size=(L,endim))
    for l in range(L):
        theta_[l,:]=theta_[l,:]/np.sqrt(np.sum(theta_[l,:]**2))
    return theta_

def oneDWassersteinV3(p,q):
    ~10 Times faster than V1

    W2=(tf.nn.top_k(tf.transpose(p),k=tf.shape(p)[0]).values-
        tf.nn.top_k(tf.transpose(q),k=tf.shape(q)[0]).values)**2

    return K.mean(W2, axis=-1)

    psort=tf.sort(p,axis=0)
    qsort=tf.sort(q,axis=0)
    pqmin=tf.minimum(K.min(psort,axis=0),K.min(qsort,axis=0))
    psort=psort-pqmin
    qsort=qsort-pqmin
    
    n_p=tf.shape(p)[0]
    n_q=tf.shape(q)[0]
    
    pcum=tf.multiply(tf.cast(tf.maximum(n_p,n_q),dtype='float32'),tf.divide(tf.cumsum(psort),tf.cast(n_p,dtype='float32')))
    qcum=tf.multiply(tf.cast(tf.maximum(n_p,n_q),dtype='float32'),tf.divide(tf.cumsum(qsort),tf.cast(n_q,dtype='float32')))
    
    indp=tf.cast(tf.floor(tf.linspace(0.,tf.cast(n_p,dtype='float32')-1.,tf.minimum(n_p,n_q)+1)),dtype='int32')
    indq=tf.cast(tf.floor(tf.linspace(0.,tf.cast(n_q,dtype='float32')-1.,tf.minimum(n_p,n_q)+1)),dtype='int32')
    
    phat=tf.gather(pcum,indp[1:],axis=0)
    phat=K.concatenate((K.expand_dims(phat[0,:],0),phat[1:,:]-phat[:-1,:]),0)
    
    qhat=tf.gather(qcum,indq[1:],axis=0)
    qhat=K.concatenate((K.expand_dims(qhat[0,:],0),qhat[1:,:]-qhat[:-1,:]),0)
          
    W2=K.mean((phat-qhat)**2,axis=0)
    return W2


def sWasserstein(P,Q,theta,nclass,Cp=None,Cq=None):
    lambda_=10.0
    p=K.dot(P,K.transpose(theta))
    q=K.dot(Q,K.transpose(theta))
    sw=lambda_*K.mean(oneDWassersteinV3(p,q))
    if (Cp is not None) and (Cq is not None):
        for i in range(nclass):
            pi=tf.gather(p,tf.squeeze(tf.where(tf.not_equal(Cp[:,i],0))))
            qi=tf.gather(q,tf.squeeze(tf.where(tf.not_equal(Cq[:,i],0))))
            sw=sw+100.*K.mean(oneDWassersteinV3(pi,qi))
    return sw







def reinitLayers(model):
    # This code reinitialize a keras/tf model
    session = K.get_session()
    for layer in model.layers: 
        if isinstance(layer, keras.engine.topology.Container):
            reinitLayers(layer)
            continue
#         print("LAYER::", layer.name)
        for v in layer.__dict__:
            v_arg = getattr(layer,v)
            if hasattr(v_arg,'initializer'):
                initializer_method = getattr(v_arg, 'initializer')
                initializer_method.run(session=session)
#                 print('reinitializing layer {}.{}'.format(layer.name, v))

def loadData(name='EO'):
    if name=='EO':
        mat_contents = sio.loadmat('Ship-Dataset/EO/EOTrain.mat' )
        temp = mat_contents['x_train']
        temp = np.array(temp, dtype = 'float32')
        x_train = temp/255.
        y_train = mat_contents['y_train']
        y_train = np.transpose(y_train)

        mat_contents = sio.loadmat('Ship-Dataset/EO/EOVal.mat')
        temp = mat_contents['x_val']
        temp = np.array(temp, dtype = 'float32')
        x_test = temp/255.
        y_test = mat_contents['y_val']
        y_test = np.transpose(y_test)
    elif name=='SAR':
        mat_contents = sio.loadmat('Ship-Dataset/SAR/SARTrain.mat')
        temp = mat_contents['x_train']
        temp = np.squeeze(temp , axis = 1)
        temp = np.array(temp, dtype = 'float32')
        x_train = 2.*(temp/255.)-1.
        x_train = np.expand_dims(x_train,3)
        y_train = mat_contents['y_train']
        y_train = np.transpose(y_train)

        mat_contents = sio.loadmat('Ship-Dataset/SAR/SARTest.mat')
        temp = mat_contents['x_test']
        temp = np.squeeze(temp , axis = 1)
        temp = np.array(temp, dtype = 'float32')
        x_test = 2.*(temp/255.)-1.
        x_test = np.expand_dims(x_test,3)
        y_test = mat_contents['y_test']
        y_test = np.transpose(y_test)
#         mat_contents = sio.loadmat('Ship-Dataset/SAR/SARVal.mat')
#         temp = mat_contents['x_val']
#         temp = np.squeeze(temp , axis = 1)
#         temp = np.array(temp, dtype = 'float32')
#         z_valid = 2.*(temp/255.)-1.
#         z_valid = np.expand_dims(z_valid,3)
#         yz_valid = mat_contents['y_val']
#         yz_valid = np.transpose(yz_valid)
    else:
        print("Invalid input")
        return 
    return x_train,keras.utils.to_categorical(y_train,2),x_test,keras.utils.to_categorical(y_test,2)
    
def randperm(X,y):
    assert X.shape[0]==y.shape[0]
    ind=np.random.permutation(X.shape[0])
    X=X[ind,...]
    y=y[ind,...]
    return X,y

def batchGenerator(label,batchsize,nofclasses=2,seed=1,noflabeledsamples=None):
    N=label.shape[0]
    if not(noflabeledsamples):
        M=int(batchsize/nofclasses)
        ind=[]
        for i in range(nofclasses):
            labelIndex=np.argwhere(label[:,i]).squeeze()
            randInd=np.random.permutation(labelIndex.shape[0])
            ind.append(labelIndex[randInd[:M]])
        ind=np.asarray(ind).reshape(-1)
         
        labelout=label[ind]
    else:
        np.random.seed(seed)
        portionlabeled=min(batchsize/2,noflabeledsamples*nofclasses)
        M=portionlabeled/nofclasses
        indsupervised=[]
        indunsupervised=np.array([])
        for i in range(nofclasses):
            labelIndex=np.argwhere(label[:,i]).squeeze()
            randInd=np.random.permutation(labelIndex.shape[0])
            indsupervised.append(labelIndex[randInd[:noflabeledsamples]])
            indunsupervised=np.append(indunsupervised,np.array(labelIndex[randInd[noflabeledsamples:]]))
        np.random.seed()
        ind=[]  
        for i in range(nofclasses):
            ind.append(np.random.permutation(indsupervised[i])[:M])
        ind=np.asarray(ind).reshape(-1)
        indunsupervised=np.random.permutation(indunsupervised)      
        
        labelout=np.zeros((nofclasses*(batchsize/nofclasses),nofclasses))
        labelout[:portionlabeled]=label[ind,:]
        ind=np.concatenate((ind,indunsupervised[:nofclasses*(batchsize/nofclasses)-ind.shape[0]]))
    return ind.astype(int),labelout


