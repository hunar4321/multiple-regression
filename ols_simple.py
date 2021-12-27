import numpy as np
import matplotlib.pylab as plt
import time

fet=29; lab=100;

x = np.random.randn(fet,lab);
bt = np.random.randn(1,fet)*10;

y = np.dot(bt, x)

## add some noise
y = y+np.random.randn(1, lab)*0

t = time.time()
pv = np.linalg.pinv(x) 
Wm = y @ pv
print("Elapsed1:", time.time()-t)

# plt.figure(1)
# plt.plot(bt[0])
# plt.plot(Wm[0])


def learn(xvec, yvec, sx):
    kapa = np.dot(xvec, yvec)/sx
    return kapa
def predict(kapa, vec):
    p = kapa * vec
    return p

tx=np.copy(x);
ty=np.copy(y);
k=1;

t=time.time()
sx=np.zeros(fet); 
b =np.zeros(fet);
for i in range(fet-1, -1, -1):
    sx[i] = np.dot(x[i,:], x[i, :])
    for j in range(0, fet-k):
        kapa1 = learn(x[j, :], x[i, :], sx[i])
        xhatj = predict(kapa1, x[i, :])
        x[j,:] = x[j, :] - xhatj
    k +=1
    
for i in range(fet-1, 0, -1): 
    kapa2 = learn(y, x[i,:], sx[i])
    yhat = predict(kapa2, x[i,:])
    y = y - yhat 
    
b[0] = learn(y, x[0,:], sx[0]) 
for i in range(fet-1):
    ty = ty - b[i]*tx[i,:]
    b[i+1] = learn(ty, x[i+1,:], sx[i+1] )
    
    
print("Elapsed2:", time.time()-t)

plt.figure(2)
plt.title("weights")
plt.plot(bt[0]); plt.plot(Wm[0]); plt.plot(b);


ypred = np.dot(b, tx)
y = np.dot(bt, tx)

plt.figure(3)
plt.title("predictiion")
plt.plot(y[0])
plt.plot(ypred)


### previous implimentations

# tx=np.copy(x);
# ty=np.copy(y);
# k=1;

# t=time.time()
# sx=np.zeros(fet); 
# b =np.zeros(fet);
# for i in range(fet-1, -1, -1):
#     sx[i] = np.dot(x[i,:], x[i, :])
#     for j in range(0, fet-k):
#         kapa1 = np.dot(x[j,:], x[i,:])/sx[i]
#         xhatj = kapa1 * x[i,:]
#         x[j,:] = x[j, :] - xhatj
#     k +=1
    
# for i in range(fet-1, 0, -1):    
#     kapa2 = np.dot(y, x[i,:])/sx[i] 
#     yhat = kapa2 * x[i,:]
#     y = y - yhat 
    
# b[0] = np.dot(y, x[0,:])/sx[0]   
# for i in range(fet-1):
#     ty = ty - b[i]*tx[i,:]
#     b[i+1] = np.dot(ty, x[i+1,:])/sx[i+1] 
    
    
# print("Elapsed2:", time.time()-t)

# plt.figure(2)
# plt.plot(bt[0]); plt.plot(Wm[0]); plt.plot(b);




# ## simple working implimentation
# ty=np.copy(y);
# tx=np.copy(x);
# yh=np.copy(ty);
# k=1;

# t=time.time()
# sx=np.zeros(fet); 
# b =np.zeros(fet);
# ## iterate from last to first (except first element)
# for i in range(fet-1, 0, -1):
#     sx[i] = np.dot(x[i,:], x[i, :])
#     kapa1 = np.dot(yh, x[i,:])/sx[i] 
#     yhat = kapa1 * x[i,:]
#     yh = yh - yhat 
#     for j in range(0, fet-k):
#         kapa2 = np.dot(x[j,:], x[i,:])/sx[i]
#         xhat = kapa2 * x[i,:]
#         x[j,:] = x[j, :] - xhat
#     k +=1

# b[0] = np.dot(yh, x[0,:])/np.dot(x[0,:], x[0,:])    
# for i in range(fet-1):
#     y = y - b[i]*tx[i,:]
#     b[i+1] = np.dot(y, x[i+1,:])/np.dot(x[i+1], x[i+1])
    
    
# print("Elapsed2:", time.time()-t)

# plt.figure(2)
# plt.plot(bt[0]); plt.plot(Wm[0]); plt.plot(b);




