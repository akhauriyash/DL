import numpy as np
import time
import matplotlib.pyplot as plt
import math
from scipy.sparse import csr_matrix
import scipy.signal
from tensorflow.examples.tutorials.mnist import input_data

def convolve_forwardpass(image, kernel):
    r, c, rk, ck = image.shape[0], image.shape[1], kernel.shape[0], kernel.shape[1]
    exi = np.empty((r-rk+1, c-ck+1))
    ktemp = np.reshape(kernel, -1)
    for rtemp in range(r-rk+1):
        for ctemp in range(c-ck+1):
            temp = 0
            o = np.reshape(i[rtemp:rtemp+rk, ctemp:ctemp+ck], -1)
            for zs in range(len(ktemp)):
                temp+=o[zs]*ktemp[8-zs]
            exi[rtemp][ctemp] = temp
    return exi
#    return scipy.signal.convolve(i, k, mode='valid')

    
def sigmoid(x, derivative = False):
    if(derivative == True):
        return x*(1-x)
    return (1/(1+np.exp(-1*x)))
    

def relu_derivative(array):
    k = array.shape
    array = np.reshape(array, -1)
    for num in range(len(array)):
        if(array[num] > 0):
            array[num] = 1
        else:
            array[num] = 0
    return np.reshape(array, (k))
    
    
def fullyconnected_forwardpass(image, hlsize, syn):
    image = image.T
    try:
        z = sigmoid(np.dot(image[None, :], syn))
    except:
        z = sigmoid(np.dot(image.T, syn))
    return z, syn

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))
    

def maxpool_forwardpass(image, pool_size = 2, stride = 1):
    r, c = image.shape[0], image.shape[1]
    out = []
    backpass = np.empty((28,28))
    backpass = np.zeros_like(backpass)
    for rtemp in range(0, r-pool_size+1, stride):
        for ctemp in range(0, c-pool_size+1, stride):
            temps = image[rtemp:rtemp+pool_size, ctemp:ctemp+pool_size]
            indices = np.where(temps == temps.max())
            if(indices[0].shape[0] > 1) or (indices[1].shape[0] > 1):
                pass
            else:
                backpass[rtemp + indices[0][0]][ctemp + indices[1][0]] = temps.max()
    for rtemp in range(0, r-pool_size+1, stride):
        for ctemp in range(0, c-pool_size+1, stride):
            out.append(np.amax(image[rtemp:rtemp+pool_size, ctemp:ctemp+pool_size]))
    return (np.reshape(out, (math.ceil((r-pool_size+1)/stride), math.ceil((c-pool_size+1)/stride)))), backpass
   
    
def flatten(image):
    return np.reshape(image, -1)
    
    
def relu(image):
    return np.maximum(image, 0, image)
    
    
def rot180(image):
    return np.rot90(np.rot90(image))

    
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
EPOCH, BATCHSIZE = 5, 128
NUM_KERNELS = 7
k1 = np.random.rand(3,3)-0.5
k2 = np.random.rand(3,3)-0.5
k3 = np.random.rand(3,3)-0.5
k4 = np.random.rand(3,3)-0.5
k5 = np.random.rand(3,3)-0.5
k6 = np.random.rand(3,3)-0.5
k7 = np.random.rand(3,3)-0.5
k1flip = rot180(k1)
k2flip = rot180(k2)
k3flip = rot180(k3)
k4flip = rot180(k4)
k5flip = rot180(k5)
k6flip = rot180(k6)
k7flip = rot180(k7)
kernels = [k1, k2, k3, k4, k5, k6, k7]
synoutput = 2*np.random.random((1372, 10))-1
r = []
c = 0
for epoch in range(EPOCH):
    for batchid in range(BATCHSIZE):
        c+=1
        maxpoolbackpass = []
        convkernelmul = []
        img = np.reshape(mnist.train.images[batchid], (28,28))
        img = np.pad(img, 1, mode = 'constant')
        i = np.asarray(img)
        feature_images=[i,i,i,i,i,i,i]
        for item in range(NUM_KERNELS):
#            print("Shape before convolution: ", feature_images[item].shape) # Shape (30,30)
            feature_images[item] = convolve_forwardpass(feature_images[item], kernels[item])
#            print("Shape after 1st convolution:", feature_images[item].shape)# Shape (28,28)
            feature_images[item] = relu(feature_images[item])
#            print(feature_images[item].shape)
            feature_images[item], append_maxpoolbackpass = maxpool_forwardpass(feature_images[item], stride = 2)
            maxpoolbackpass.append(append_maxpoolbackpass)
#            print("Shape after maxpool: ", feature_images[item].shape)  # Shape (14,14)
        z = flatten(feature_images)
        flat = z
        
        z, synoutput = fullyconnected_forwardpass(z, 10, synoutput)
        z = softmax(z)

        outputerror = 0.5*np.sum(np.square(mnist.train.labels[batchid]-z))
        
        outputdelta = outputerror*sigmoid(z, True)
        
        outputdelta = outputdelta.T
        
        synoutput += np.dot(z, outputdelta) ## Shape multiplication as: (1,10) (10,1) ==> (1)
        
#        flattenerror = outputdelta.dot(flat.T[None, :]) ## Shape multiplication as (10,1) (1, 14x14x7) 
        #FLAT ERROR WILL DIRECTLY PROPOGATE BACK, maintaining shape of (14x14x7)
        flat = np.reshape(flat, (NUM_KERNELS, 14, 14))
        
        flatbackprop = np.zeros_like((NUM_KERNELS, 28, 28))
        for num in range(NUM_KERNELS):
            convkernelmul.append(relu_derivative(maxpoolbackpass[num]))
                    
        for zoomba in range(NUM_KERNELS):
            convkernelmul[zoomba] = np.pad(convkernelmul[zoomba], 1, mode='constant')
#            plt.imshow(convkernelmul[zoomba])
#            plt.show()
#            print(convkernelmul[zoomba].shape)
#            
        
#        print("Flat, then maxpoolbackpass shapes: ")
#        print(flat.shape)
#        print(len(maxpoolbackpass))
#        for zoomba in range(NUM_KERNELS):
#            plt.imshow(maxpoolbackpass[zoomba])
#            plt.show()
#        time.sleep(10)


#        print("Error: ")
        print(c)
        r.append((outputerror, c))
r = np.asarray(r)
print(r.shape)
r = r.T
plt.plot(r[1], r[0])
plt.axis([0, 620, 0, 1])
plt.show()
        
            

#k =  [[0.2056752, -0.08870374, 0.26896033], [-0.47955601, -0.36336509, -0.07082405], [0.23903859, -0.47164089, 0.19942926]]
#k = np.asarray(k)
#print(k.shape)
#for iters in range(128):
#    img = np.reshape(mnist.train.images[iters], (28,28))
#    img = np.pad(img, 1, mode = 'constant')
#    i = np.asarray(img)
#    z = convolve_forwardpass(i, k)
#    z = relu(z)
#    z = maxpool_forwardpass(z, stride = 2)
#    plt.imshow(z)
#    plt.show()
