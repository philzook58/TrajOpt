import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim
from scipy import linalg
batch = 1
N = 3
T = 10.0
dt = T/N
NVars = 4
NControls = 1
def getNewState():
	#we 're going to also pack f into here
	x = torch.ones(batch,N,NVars+NControls, requires_grad=True) 

	#f = torch.zeros(batch, N-1, requires_grad=True) 

	l = torch.zeros(batch, N-1,NVars, requires_grad=False) 
	return x, l

def calc_loss(x, l ,rho): # l,
	#depack f, it has one less time point 
	f = x[:,:-1,NVars]
	leftoverf = x[:,-1,NVars]
	x = x[:,:,:-1]

	delx = (x[:,1:,:] - x[:, :-1,:]) / dt

	xbar = (x[:,1:,:] + x[:, :-1,:]) / 2
	dxdt = torch.zeros(batch, N-1,NVars)
	THETA = 2
	THETADOT = 3
	X = 0
	V = 1
	dxdt[:,:,X] = xbar[:,:,V]
	dxdt[:,:,V] = f
	dxdt[:,:,THETA] = xbar[:,:,THETADOT] 
	dxdt[:,:,THETADOT] = -torch.sin(xbar[:,:,THETA]) + f*torch.cos(xbar[:,:,THETA])




	xres = delx - dxdt

	lagrange_mult = torch.sum(l * xres, dim=1).sum(1)

	#cost =  1.0*torch.sum(torch.abs(x[:,:,THETA]-np.pi), dim=1) 
	cost = torch.sum((x+1)**2+(x+1)**2, dim=0).sum(0).sum(0)
	cost += torch.sum((f+1)**2, dim=0).sum(0).sum(0)
	cost += 1
	penalty = rho*torch.sum( xres**2 , dim=1).sum(1) 
	#cost =  1.0*torch.sum(torch.abs(x[:,:,THETA]-np.pi), dim=1) 
	#cost += 0.5*torch.sum( f**2, dim=1)

	#cost += 0.1*torch.sum(-torch.log(xbar[:,:,X] + 1) - torch.log(1 - xbar[:,:,X]),dim=1)
	#cost += 0.1*torch.sum(-torch.log(xbar[:,:,V] + 1) - torch.log(1 - xbar[:,:,V]),dim=1)
	cost += (leftoverf**2).sum(0)




	#total_cost =  cost + lagrange_mult + penalty 

	return cost, penalty, lagrange_mult,  xres

def getFullHess(): #for experimentation
	pass

def getGradHessBand(loss, B, x):
	#B = bandn
	delL0, = torch.autograd.grad(loss, x, create_graph=True)
	delL = delL0[:,:].view(-1) #remove x0
	#print("del ", delL)
	hess = torch.zeros(B,N,NVars+NControls, requires_grad=False).view(B,-1)
	for i in range(B):
		y = torch.zeros(N,NVars+NControls, requires_grad=False).view(-1)
		y[i::B]=1
		#print(y)
		#print(y.shape)
		#rint(delL.shape)
		delLy = torch.dot(delL , y)
		#print(delLy)
		delLy.backward(retain_graph=True) #(i != B-1)
		#print(hess.shape)
		#print(x.grad.shape)
		hess[i,:] = x.grad[:,:,:].view(-1) #also remove x0
		#print(hess[i,:])
		#print(x.grad) 
		x.grad.data.zero_()
	#print(hess)
	nphess = hess.detach().numpy()
	#print(nphess)
	#print(nphess)
	for i in range(B):
		nphess[:,i::B] = np.roll(nphess[:,i::B], -i+B//2, axis=0)
	print(nphess)
	#hessband = removeX0(nphess[:B//2+1,:])
	grad = removeX0(delL.detach().numpy())
	return grad, nphess #hessband

def removeX0(x):
	return x.reshape(-1,N,NVars+NControls)[:,1:,:].reshape(-1, (N-1)*(NVars+NControls))



x, l = getNewState()
rho = 0.1
for j in range(15):
	while True:
		cost, penalty, lagrange_mult, xres = calc_loss(x, l, rho)
		#print(total_cost)
		print("hey now")
		#print(cost)
		#total_cost = cost + penalty + lagrange_mult
		total_cost = cost
		gradL, hess = getGradHessBand(total_cost, (NVars+NControls)*3, x)
		#print(hess)
		#print(hess.shape)
		gradL = gradL.reshape(-1)
		#print(gradL.shape)

		#easiest thing might be to put lagrange mutlipleirs into x.
		#Alternatively, use second order step in penalty method.
		dx = linalg.solve_banded((7,7), hess, gradL) #lower=True
		newton_dec = np.dot(dx,gradL)
		dx = dx.reshape(-1,N-1,NVars+NControls)
		with torch.no_grad():
			x[:,1:,:] -= torch.tensor(dx)
			costval = cost.detach().numpy()
			if newton_dec/costval < 1e-5:
				break
		#print(x)
	with torch.no_grad():
		l += 2 * rho * xres
	rho = rho + 0.1
print(x)
