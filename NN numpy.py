import numpy as np
#Returnes the sigmoid function. If derive = True,
#then returns the derivative at the point.
def sigfunc(x, derivative=False):
    if (derivative == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-1* x))
X = np.array([[0,0,0],[0,0,1],[0,1,0],
              [1,0,0],[0,1,1],[1,0,1],
              [1,1,0],[1,1,1]])
y = np.array([[1,0,1,1,0,0,1,0]]).T
hl1size = 4
hl2size = 5
syn0 = 2 * np.random.random((3, hl1size)) - 1
syn1 = 2 * np.random.random((hl1size, hl2size)) - 1
syn2 = 2 * np.random.random((hl2size, 1)) - 1
for k in range(50000): #Number of iterations to run
    l0 = X
    l1 = sigfunc(np.dot(l0, syn0))
    l2 = sigfunc(np.dot(l1, syn1))
    l3 = sigfunc(np.dot(l2, syn2))
    l3error = y - l3
    l3delta = l3error * sigfunc(l3, True)
    l2error = l3delta.dot(syn2.T)
    l2delta = l2error*sigfunc(l2, True)
    l1error = l2delta.dot(syn1.T)
    l1delta = l1error*sigfunc(l1, True)
#    if(k%10000 == 0):
#        print("\n ==============l3delta: \n ")
#        print(l3delta)
#        print("\n ==============l2delta: \n ")
#        print(np.around(l2delta, decimals = 4))
#        print("\n ==============l1delta: \n ")
#        print(np.around(l1delta, decimals = 4))
#        print("\n =================== \n===============================\n ==========\n")
    syn0 += np.dot(l0.T, l1delta)
    syn1 += np.dot(l1.T, l2delta)
    syn2 += np.dot(l2.T, l3delta)
    
print(l3)
#Put your input here
l0 = np.array([[1,0,0]])
#Runs l0 through the adjusted neural network
l1 = sigfunc(np.dot(l0, syn0))
l2 = sigfunc(np.dot(l1, syn1))
l3 = sigfunc(np.dot(l2, syn2))
#provides prediction result
print("Prediction result: ")
print(l3[0][0])

'''   
print('\n=====================================\n')
print("X: 8x3")
print(X)
print("\n ============ \n ")
print("l1: 8x" + str(hl1size))
print(np.around(l1, decimals = 2))
print(np.around(syn0, decimals = 2))
print("\n ============ \n ")
print("l2: 8x" + str(hl2size))
print(np.around(l2, decimals = 2))
print(np.around(syn1, decimals = 2))

print("\n ============ \n ")
print("l3: 8x1")
print(np.around(l3, decimals = 6))
print(np.around(syn2, decimals = 6))

'''


