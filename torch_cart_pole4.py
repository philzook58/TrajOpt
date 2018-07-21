import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim
batch = 1
N = 50
T = 10.0
dt = T/N
NVars = 4


def getNewState():

	x = torch.zeros(batch,N,NVars, requires_grad=True) 

	f = torch.zeros(batch, N-1, requires_grad=True) 

	l = torch.zeros(batch, N-1,NVars, requires_grad=False) 
	return x,f,l







def calc_loss(x,f, l, rho):
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

	#dyn_err = torch.sum(torch.abs(xres) + torch.abs(vres), dim=1) #torch.sum(xres**2 + vres**2, dim=1) # + Abs of same thing?

	lagrange_mult = torch.sum(l * xres, dim=1).sum(1)


	#cost = 0
	cost =  1.0*torch.sum(torch.abs(x[:,:,THETA]-np.pi), dim=1) # 0.1*torch.sum((x[:,:,X]-1)**2, dim=1)  + 
	#cost += 2.0 * torch.sum((x[:,30:-1,THETA] - np.pi)**2,dim=1)
	#cost += 7.0*torch.sum( torch.abs(xres)+ xres**2 , dim=1).sum(1) 
	penalty = rho*torch.sum( xres**2 , dim=1).sum(1) 
	#  + 1*torch.sum( torch.abs(xres)+ xres**2 , dim=1).sum(1) 
	# 5.0*torch.sum( torch.abs(xres)+ xres**2 , dim=1).sum(1) +
	cost += 0.01*torch.sum( f**2, dim=1)
	#cost += torch.sum(-torch.log(f + 1) - torch.log(1 - f),dim=1)
	cost += 0.1*torch.sum(-torch.log(xbar[:,:,X] + 1) - torch.log(1 - xbar[:,:,X]),dim=1)
	cost += 0.1*torch.sum(-torch.log(xbar[:,:,V] + 1) - torch.log(1 - xbar[:,:,V]),dim=1)
	
	#cost += torch.sum(-torch.log(xres + .5) - torch.log(.5 - xres),dim=1).sum(1)
	
	# torch.sum( torch.abs(xres), dim=1).sum(1)*dt + 
	#cost = torch.sum((x-1)**2, dim=1)


	total_cost =   cost + lagrange_mult + penalty  #100 * dyn_err + reward

	return total_cost, lagrange_mult, cost, xres



import torch.optim as optim




learning_rate = 0.001



x, f, l = getNewState()
optimizers = [torch.optim.SGD([x,f], lr= learning_rate),
	 torch.optim.Adam([x,f]),
	 torch.optim.Adagrad([x,f])]
optimizerNames = ["SGD", "Adam", "Adagrad"]
optimizer = optimizers[1]
#optimizer = torch.optim.SGD([x,f], lr= learning_rate)
#optimizer = torch.optim.Adam([x,f])
#optimizer = torch.optim.Adagrad([x,f])
costs= []
path = []
mults = []
rho = 0.1
prev_cost = 0
for j in range(15):
	prev_cost = None
	for i in range(1,10000):

		total_cost, lagrange, cost, xres = calc_loss(x,f, l, rho)

		costs.append(total_cost[0])
		if i % 5 == 0:
			#pass
			print(total_cost)
		optimizer.zero_grad()


		total_cost.backward()


		optimizer.step()
		
		with torch.no_grad():
			x[0,0,2] = 0#np.pi+0.3 # Initial Conditions
			x[0,0,0] = 0
			x[0,0,1] = 0
			x[0,0,3] = 0
			#print(x.grad.norm()/N)
			#print((x.grad.norm()/total_cost/N).detach().numpy() < 0.01)
			#if (x.grad.norm()).detach().numpy()/N < 0.1: #Put Convergence condition here
		
			if i > 2000:
				break	
			if prev_cost:
				if ((total_cost - prev_cost).abs()/total_cost).detach().numpy() < 0.000001:
					pass #break

			prev_cost = total_cost

		
	total_cost, lagrange, cost, xres = calc_loss(x,f, l, rho)
	costs.append(total_cost[0])

	
	with torch.no_grad():
		l += 2 * rho * xres

	rho = rho + 0.5
	print(rho)

plt.subplot(131)
plt.plot(xres[0,:,0].detach().numpy(), label='Xres')
plt.plot(xres[0,:,1].detach().numpy(), label='Vres')
plt.plot(xres[0,:,2].detach().numpy(), label='THeres')
plt.plot(xres[0,:,3].detach().numpy(), label='Thetadotres')

plt.legend(loc='upper right')
#plt.figure()
plt.subplot(132)
plt.plot(x[0,:,0].detach().numpy(), label='X')
plt.plot(x[0,:,1].detach().numpy(), label='V')
plt.plot(x[0,:,2].detach().numpy(), label='Theta')
plt.plot(x[0,:,3].detach().numpy(), label='Thetadot')
plt.plot(f[0,:].detach().numpy(), label='F')
#plt.plot(cost[0,:].detach().numpy(), label='F')
plt.legend(loc='upper right')
#plt.figure()
plt.subplot(133)
plt.plot(costs)


plt.show()