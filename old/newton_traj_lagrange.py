import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim
from scipy import linalg

N = 50
T = 10.0
dt = T/N
NVars = 4
NControls = 1
batch = (2*NVars+NControls)*3
def getNewState():
	#we 're going to also pack f into here
	x = torch.zeros(batch,N,2*NVars+NControls, requires_grad=True) 

	#f = torch.zeros(batch, N-1, requires_grad=True) 

	#l = torch.zeros(batch, N-1,NVars, requires_grad=False) 
	return x# , l

def calc_loss(x, rho, prox=0): # l,
	#depack f, it has one less time point 
	cost = 0.1*torch.sum(x**2)
	cost += prox * torch.sum((x - x.detach())**2)
	#lagrange first
	
	f = x[:,1:,NVars:NControls+NVars]
	l = x[:,1:,:NVars]
	x = x[:,:,NControls + NVars:]
	'''
	#lagrage last
	f = x[:,1:,:NControls]
	l = x[:,1:,NControls + NVars:]
	x = x[:,:,NControls:NControls + NVars]
	#leftoverf = x[:,0,:NControls]
	'''
	#cost += 1.0*torch.sum(x**2)
	#cost = 0.1*torch.sum(x)

	delx = (x[:,1:,:] - x[:, :-1,:]) / dt

	xbar = (x[:,1:,:] + x[:, :-1,:]) / 2
	dxdt = torch.zeros(batch, N-1,NVars)
	THETA = 2
	THETADOT = 3
	X = 0
	V = 1
	dxdt[:,:,X] = xbar[:,:,V]
	#print(dxdt.shape)
	#print(f.shape)
	dxdt[:,:,V] = f[:,:,0]
	dxdt[:,:,THETA] = xbar[:,:,THETADOT] 
	dxdt[:,:,THETADOT] = -torch.sin(xbar[:,:,THETA]) + f[:,:,0]*torch.cos(xbar[:,:,THETA])




	xres = delx - dxdt

	lagrange_mult = torch.sum(l * xres)


	#cost = torch.sum((x+1)**2+(x+1)**2, dim=0).sum(0).sum(0)
	#cost += torch.sum((f+1)**2, dim=0).sum(0).sum(0)
	#cost += 1
	penalty = rho*torch.sum( xres**2)

	
	#cost +=  1.0*torch.sum((abs(x[:,:,THETA]-np.pi)), dim=1) 
	#cost =  1.0*torch.sum((x[:,:,:]-np.pi)**2 )
	#cost += 1.0*torch.sum(x[:,:,X]**2) + 1.0*torch.sum(x[:,:,THETADOT]**2) + 1.0*torch.sum(x[:,:,V]**2)
	cost +=  1.0*torch.sum((x[:,:,THETA]-np.pi)**2 * torch.arange(N) / N )
	cost += 0.5*torch.sum(f**2)
	#cost =  1.0*torch.sum((x[:,:,:]-np.pi)**2 )
	#cost = cost1 + 1.0
	#cost += 0.01*torch.sum(x**2, dim=1).sum(0).sum(0)
	#xlim = 3
	#cost += 0.1*torch.sum(-torch.log(xbar[:,:,X] + xlim) - torch.log(xlim - xbar[:,:,X]))
	#cost += 0.1*torch.sum(-torch.log(xbar[:,:,V] + 1) - torch.log(1 - xbar[:,:,V]),dim=1)
	#cost += (leftoverf**2).sum()




	#total_cost =  cost + lagrange_mult + penalty 

	return cost, penalty, lagrange_mult,  xres

def getFullHess(): #for experimentation
	pass

def getGradHessBand(loss, B, x):
	#B = bandn
	delL0, = torch.autograd.grad(loss, x, create_graph=True)
	delL = delL0[:,1:,:].view(B,-1) #remove x0
	print("del ", delL[:,:10])
	#hess = torch.zeros(B,N-1,NVars+NControls, requires_grad=False).view(B,B,-1)
	y = torch.zeros(B,N-1,2*NVars+NControls, requires_grad=False).view(B,-1)
	
	#y = torch.eye(B).view(B,1,B)
	#print(y.shape)
	for i in range(B):
		#y = torch.zeros(N-1,NVars+NControls, requires_grad=False).view(-1)
		y[i,i::B]=1
	#print(y[:,:2*B])
	print(y.shape)
	print(delL.shape)
	delLy = torch.sum(delL * y)
	#print(delLy)
	
	
	delLy.backward() #(i != B-1)
	#torch.autograd.grad(loss, x, create_graph=True)
	#print(hess.shape)
	#print(x.grad.shape)
	#hess[i,:] = x.grad[:,1:,:].view(-1) #also remove x0
	#print(hess[i,:])
	#print(x.grad) 
	
	#print(hess)
	nphess = x.grad[:,1:,:].view(B,-1).detach().numpy()# .view(-1)# hess.detach().numpy()
	#print(nphess[:,:4])
	#print(nphess)
	for i in range(B):
		nphess[:,i::B] = np.roll(nphess[:,i::B], -i+B//2, axis=0)
	print(nphess[:,:4])
	#hessband = removeX0(nphess[:B//2+1,:])
	#grad = removeX0(delL.detach().numpy())
	return delL.detach().numpy()[0,:], nphess #hessband

x = getNewState()
rho = 7.0
prox = 1.0

while True:
	try:
		cost, penalty, lagrange_mult, xres = calc_loss(x, rho, prox)
		#print(total_cost)
		print("hey now")
		#print(cost)
		total_cost = cost + lagrange_mult + penalty
		#total_cost = cost
		gradL, hess = getGradHessBand(total_cost, (2*NVars+NControls)*3, x)
		#print(hess)
		#print(hess.shape)
		gradL = gradL.reshape(-1)
		#print(gradL.shape)

		#easiest thing might be to put lagrange mutlipleirs into x.
		#Alternatively, use second order step in penalty method.
		bandn = (2*NVars+NControls)*3//2
		print(hess.shape)
		print(gradL.shape)
		dx = linalg.solve_banded((bandn,bandn), hess, gradL) #
		x.grad.data.zero_()
		#print(hess)
		#print(hess[bandn:,:])
		#dx = linalg.solveh_banded(hess[:bandn+1,:], gradL, overwrite_ab=True)
		newton_dec = np.dot(dx,gradL)
		#df0 = dx[:NControls].reshape(-1,NControls)
		dx = dx.reshape(1,N-1,2*NVars+NControls)

		
		with torch.no_grad():
			x[:,1:,:] -= torch.tensor(dx)
			print(x[:,:5,:])
			#print(x[:,0,NVars:].shape)
			#print(df0.shape)
			costval = cost.detach().numpy()
			#break
			if newton_dec < 1e-10*costval:
				prox = 0 #prox / 4
				rho = rho * 2
			if rho > 400:
				break
		#prox = prox*0.95
	except np.linalg.LinAlgError:
		print("LINALGERROR")
		prox = prox * 2
		#break
	#print(x)

print(x)
#plt.subplot(131)
plt.plot(xres[0,:,0].detach().numpy(), label='Xres')
plt.plot(xres[0,:,1].detach().numpy(), label='Vres')
plt.plot(xres[0,:,2].detach().numpy(), label='THeres')
plt.plot(xres[0,:,3].detach().numpy(), label='Thetadotres')

plt.legend(loc='upper right')
plt.figure()
#plt.subplot(132)

plt.plot(x[0,:,1].detach().numpy(), label='X')
plt.plot(x[0,:,2].detach().numpy(), label='V')
plt.plot(x[0,:,3].detach().numpy(), label='Theta')
plt.plot(x[0,:,4].detach().numpy(), label='Thetadot')
plt.plot(x[0,:,0].detach().numpy(), label='F')




#plt.plot(cost[0,:].detach().numpy(), label='F')
plt.legend(loc='upper right')



plt.figure()
plt.plot(x[0,:,5].detach().numpy(), label='lX')
plt.plot(x[0,:,6].detach().numpy(), label='lV')
plt.plot(x[0,:,7].detach().numpy(), label='lTheta')
plt.plot(x[0,:,8].detach().numpy(), label='lThetadot')
plt.legend(loc='upper right')
#plt.subplot(133)
#plt.plot(costs)


plt.show()
