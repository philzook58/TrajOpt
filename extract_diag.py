

import numpy as np

N = 6
B = 5
h = np.diag(np.random.randn(N))
h = h + h.T
print(h)
band = y = np.zeros((B, N)) 
for i in range(B):
	y = np.zeros(N) 
	y[i::B]=1
	band[i,:] = h @ y
print(band)
for i in range(B):
	band[:,i::B] = np.roll(band[:,i::B], -i, axis=0) #B//2

print(band)
print(band[:B//2+1,:])