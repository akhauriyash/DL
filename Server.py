import socket, sys, numpy as np
def printf(format, *args):
    sys.stdout.write(format % args)
def sigfunc(x, derivative=False):
    if (derivative == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-1* x))
host = 'localhost'
port = 44211
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind((host,port))
except Exception as e:
    print(str(e))
s.listen(1)
while 1:
    conn, addr = s.accept()
    print("Connection established with: ", addr[0]," : ",  str(addr[1]))
    arrX = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],
           [0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    arrY = [[0],[0],[0],[0],[0],[0],[0],
            [0]]
    arrtest = [[0,0,0]]
    X = np.asarray(arrX,  dtype = int)
    y = np.asarray(arrY, dtype = int)
    testcase = np.asarray(arrtest, dtype = int)

    for i in range(8):
        for j in range(3):
            data = conn.recv(4)
            X[i][j] = int(data.decode())
    #RECIEVE TEST CASE:
    for j in range(3):
        data = conn.recv(4)
        testcase[0][j] = int(data.decode())
        print(testcase[0][j])
    for i in range(8):
        data = conn.recv(4)
        y[i][0] = int(data.decode())
        print(y[i][0])

    print(X)
    print(y)
    print(testcase)
    conn.send(str.encode('Matrix received @ Server\n'))
    hl1size = 4
    hl2size = 5
    syn0 = 2 * np.random.random((3, hl1size)) - 1
    syn1 = 2 * np.random.random((hl1size, hl2size)) - 1
    syn2 = 2 * np.random.random((hl2size, 1)) - 1
    for k in range(200): #Number of iterations to run
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
        syn0 += np.dot(l0.T, l1delta)
        syn1 += np.dot(l1.T, l2delta)
        syn2 += np.dot(l2.T, l3delta)
        if(k%5000 == 0):
            print("still training")

    print(l3)
    #Put your input here
    l0 = np.array(testcase)
    #Runs l0 through the adjusted neural network
    l1 = sigfunc(np.dot(l0, syn0))
    l2 = sigfunc(np.dot(l1, syn1))
    l3 = sigfunc(np.dot(l2, syn2))
    #provides prediction result
    print("Prediction result: ")
    print(l3[0][0])
    string = str(l3[0][0]) + '\n'
    conn.send(str.encode(string))
    print("Sent prediction to client\n")
    s.close()
