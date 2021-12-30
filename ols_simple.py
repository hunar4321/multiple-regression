import numpy as np
import matplotlib.pylab as plt
import time

#number of features (fet) and the number of labels (100)
fet=29; lab=100;

# generating time series data
x = np.random.randn(fet,lab);
bt = np.random.randn(1,fet)*10;

y = np.dot(bt, x)

## adding some noise
y = y+np.random.randn(1, lab)*0

print("solving using pinv")
t = time.time()
pv = np.linalg.pinv(x) 
Wm = y @ pv
print("Elapsed1:", time.time()-t)

# plt.figure(1)
# plt.plot(bt[0])
# plt.plot(Wm[0])

print("solving using the simple method of regressing the errors out from each variable")
t=time.time()
tx=np.copy(x);
ty=np.copy(y);
k=1;
sx=np.zeros(fet); 
b =np.zeros(fet);
for i in range(fet-1, -1, -1):
    sx[i] = np.dot(x[i,:], x[i, :])
    for j in range(0, fet-k):
        kapa = np.dot(x[j, :], x[i, :])/sx[i]
        xhatj = kapa * x[i,:]
        x[j,:] = x[j, :] - xhatj
    k +=1
    
for i in range(fet-1, 0, -1): 
    kapa = np.dot(y, x[i, :])/sx[i]
    yhat = kapa * x[i,:]
    y = y - yhat 
    
b[0] = np.dot(y, x[0,:])/sx[0]
for i in range(fet-1):
    ty = ty - b[i]*tx[i,:]
    b[i+1] = np.dot(ty, x[i+1,:])/sx[i+1]
    
print("Elapsed2:", time.time()-t)

# Comparing the results with the true values
plt.figure(2)
plt.title("weights")
plt.plot(bt[0]); plt.plot(Wm[0]); plt.plot(b);


ypred = np.dot(b, tx)
y = np.dot(bt, tx)

plt.figure(3)
plt.title("predictiion")
plt.plot(y[0])
plt.plot(ypred)









