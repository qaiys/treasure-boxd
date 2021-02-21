print("ok")
import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
import random
import matplotlib.pyplot as plt
print("imports")
data = sparse.random(150,150,density=0.5).toarray() #thisll be changed to be user data later on
Q = np.random.rand(2,150)
R = np.random.rand(150,2)
K = np.nonzero(data)
go = np.matmul(R,Q)
print("ready...")

nums = []
f = 150
for i in range(f):
    n = random.randint(0,len(K[0])-1)
    j = K[1][n]
    i = K[0][n]
    a = 0.002 #step size
    Q = Q - a*(data[i][j] - np.matmul(Q.T[i],R[j]) * (R.T))
    R = R - a*((data[i][j] - np.matmul(Q.T[i],R[j]) * (Q)).T)
    go = np.matmul(R,Q)
    nums.append(mean_squared_error(data,go))

print(go)
plt.plot(np.arange(0,f),nums,'o')
plt.show()
