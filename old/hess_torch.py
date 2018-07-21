import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim
from scipy import linalg
import matplotlib.pyplot as plt

N = 12

x = torch.zeros(N, requires_grad=True) 


L = torch.sum((x[1:] - x[ :-1])**2)/2 + x[0]**2/2 + x[-1]**2/2 #torch.sum((x**2))

#L.backward()
B = 3

delL, = torch.autograd.grad(L, x, create_graph=True)
print(delL)
print(x.grad)

hess = torch.zeros(B,N, requires_grad=False)
for i in range(B):
	y = torch.zeros(N, requires_grad=False) 
	y[i::B]=1
	delLy = delL @ y
	#delLy._zero_grad()

	delLy.backward(retain_graph=True)
	hess[i,:] = x.grad
	print(x.grad) 
	x.grad.data.zero_()
print(hess)
nphess = hess.detach().numpy()
print(nphess)
for i in range(B):
	nphess[:,i::B] = np.roll(nphess[:,i::B], -i, axis=0)
hessband = nphess[:B//2+1,:]

print(nphess)
print(nphess[:B//2+1,:])

b = np.zeros(N)
b[4]=1
x = linalg.solveh_banded(hessband, b, lower=True)
print(x)

plt.plot(x)
plt.show()
