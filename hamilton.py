import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import grad


batch = 1
N = 50
T = 7.0
dt = T/N
NVars = 1

x = torch.zeros(batch,N,NVars, requires_grad=True)
p = torch.zeros(batch,N,NVars, requires_grad=True)


hamiltonian = torch.sum(p**2 + x**2, dim=1)

dHdx, dHdp = grad(hamiltonian, [x,p], create_graph=True)



#print(x)

#v = torch.zeros(batch, N, requires_grad=True)

f = torch.zeros(batch, N-1, requires_grad=True)

l = torch.zeros(batch, N-1,NVars, requires_grad=True)
#with torch.no_grad():
#	x[0,0,2] = np.pi


'''
class Vars():
	def __init__(self, N=10):
		self.data = torch.zeros(batch, N, 2)
		self.data1 = torch.zeros(batch, N-1, 3)
		self.lx = self.data1[:,:,0]
		self.lv = self.data1[:,:,1]
		self.f = self.data1[:,:,2]
		self.x = self.data[:,:,0]
		self.v = self.data[:,:,1]
'''
def calc_loss(x,f, l):
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
	dxdt[:,:,THETADOT] = -torch.sin(xbar[:,:,THETA]) + torch.cos(f)




	xres = delx - dxdt

	#dyn_err = torch.sum(torch.abs(xres) + torch.abs(vres), dim=1) #torch.sum(xres**2 + vres**2, dim=1) # + Abs of same thing?

	lagrange_mult = torch.sum(l * xres, dim=1).sum(1)


	cost = 0.1*torch.sum((x[:,:,X]-1)**2, dim=1)  + .1*torch.sum( f**2, dim=1) + torch.sum( torch.abs(xres), dim=1).sum(1)*dt +  torch.sum((x[:,:,THETA]-np.pi)**2, dim=1)*0.01 

	#cost = torch.sum((x-1)**2, dim=1)


	total_cost =   cost + lagrange_mult  #100 * dyn_err + reward

	return total_cost, lagrange_mult, cost, xres

#print(x.grad)
#print(v.grad)
#print(f.grad)

import torch.optim as optim
'''
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
'''

#Could interleave an ODE solve step - stinks that I have to write dyanmics twice
#Or interleave a sepearate dynamic solving
# Or could use an adaptive line search. Backtracking
# Goal is to get dyn_err quite small


'''
learning_rate = 0.001
for i in range(40):
	total_cost=calc_loss(x,v,f)
	#total_cost.zero_grad()
	total_cost.backward()
	while dyn_loss > 0.01:
		dyn_loss.backward()
		with torch.no_grad():
			learning_rate = dyn_loss / (torch.norm(x.grad[:,1:]) + (torch.norm(v.grad[:,1:])
			x[:,1:] -= learning_rate * x.grad[:,1:] # Do not change Starting conditions
			v[:,1:] -= learning_rate * v.grad[:,1:]
	reward.backward()
	with torch.no_grad():
		f -= learning_rate * f.grad
'''
learning_rate = 0.05
costs= []
for i in range(4000):
	total_cost, lagrange, cost, xres = calc_loss(x,f, l)
	costs.append(total_cost[0])
	print(total_cost)
	#print(x)
	#total_cost.zero_grad()

	total_cost.backward()
	with torch.no_grad():
		#print(f.grad)
		#print(lx.grad)
		#print(x.grad)
		#print(v.grad)
		f -= learning_rate * f.grad
		#l += learning_rate * l.grad

		#print(x.grad[:,1:])
		x[:,1:,:] -= learning_rate * x.grad[:,1:,:] # Do not change Starting conditions

	#x.grad.data.zero_()

	#f.grad.data.zero_()
	l.grad.data.zero_()
	total_cost, lagrange, cost, xres = calc_loss(x,f, l)
	costs.append(total_cost[0])
	print(total_cost)
	#print(x)
	#total_cost.zero_grad()

	total_cost.backward()
	with torch.no_grad():
		#print(f.grad)
		#print(lx.grad)
		#print(x.grad)
		#print(v.grad)
		#f -= learning_rate * f.grad
		l += learning_rate * l.grad

		#print(x.grad[:,1:])
		#x[:,1:,:] -= learning_rate * x.grad[:,1:,:] # Do not change Starting conditions

	x.grad.data.zero_()

	f.grad.data.zero_()
	#l.grad.data.zero_()


print(x)
#print(v)
print(f)

plt.plot(xres[0,:,0].detach().numpy(), label='Xres')
plt.plot(xres[0,:,1].detach().numpy(), label='Vres')
plt.plot(xres[0,:,2].detach().numpy(), label='THeres')
plt.plot(xres[0,:,3].detach().numpy(), label='Thetadotres')
plt.plot(x[0,:,0].detach().numpy(), label='X')
plt.plot(x[0,:,1].detach().numpy(), label='V')
plt.plot(x[0,:,2].detach().numpy(), label='Theta')
plt.plot(x[0,:,3].detach().numpy(), label='Thetadot')
plt.plot(f[0,:].detach().numpy(), label='F')

#plt.plot(costs)
#plt.plot(l[0,:,0].detach().numpy(), label='Lx')

plt.legend(loc='upper left')
plt.show()