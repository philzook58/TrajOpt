import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim
from scipy import linalg
import time

N = 100
T = 10.0
dt = T/N
NVars = 4
NControls = 1
# Enum values
X = 0
V = 1
THETA = 2
THETADOT = 3

#The bandwidth number for solve_banded

bandn = (NVars+NControls)*3//2
# We will use this many batches so we can get the entire hessian in one pass
batch = bandn * 2 + 1


def getNewState():
	#we 're going to also pack f into here
	#The forces have to come first for a good variable ordering the the hessian
	x = torch.zeros(batch,N,NVars+NControls, requires_grad=True) 
	l = torch.zeros(1, N-1,NVars, requires_grad=False) 
	return x, l

#Compute the residual with respect to the dynamics
def dynamical_res(x):
	f = x[:,1:,:NControls]
	x = x[:,:,NControls:]

	delx = (x[:,1:,:] - x[:, :-1,:]) / dt

	xbar = (x[:,1:,:] + x[:, :-1,:]) / 2
	#dxdt = torch.zeros(x.shape[0], N-1,NVars)
	dxdt = torch.zeros_like(xbar)
	dxdt[:,:,X] = xbar[:,:,V]
	dxdt[:,:,V] = f[:,:,0]
	dxdt[:,:,THETA] = xbar[:,:,THETADOT] 
	dxdt[:,:,THETADOT] = -torch.sin(xbar[:,:,THETA]) + f[:,:,0]*torch.cos(xbar[:,:,THETA])

	xres = delx - dxdt
	return xres

def calc_loss(x, l, rho):
	xres = dynamical_res(x)
	# Some regularization. This encodes sort of that all variables -100 < x< 100
	cost = 0.1*torch.sum(x**2)
	# The forces have to come first for a good variable ordering the the hessian
	f = x[:,1:,:NControls]
	x = x[:,:,NControls:]

	lagrange_mult = torch.sum(l * xres)
	penalty = rho*torch.sum(xres**2)

	
	#Absolute Value craps it's pants unfortunately.
	#I tried to weight it so it doesn't feel bad about needing to swing up
	cost +=  1.0*torch.sum((x[:,:,THETA]-np.pi)**2 * torch.arange(N) / N )
	cost += 0.5*torch.sum(f**2)
	xlim = 0.4
	#Some options to try for inequality constraints. YMMV.
	#cost += rho*torch.sum(-torch.log(xbar[:,:,X] + xlim) - torch.log(xlim - xbar[:,:,X]))
	#The softer inequality constraint seems to work better.
	# the log loses it's mind pretty easily
	# tried adding ln rho in there to make it harsher as time goes on?
	#cost += torch.sum(torch.exp((-xbar[:,:,X] - xlim)*(5+np.log(rho+0.1))) + torch.exp((xbar[:,:,X]- xlim)*(5+np.log(rho+0.1))))
	#Next one doesn't work?
	#cost += torch.sum(torch.exp((-xbar[:,:,X] - xlim)) + torch.exp((xbar[:,:,X]- xlim)))**(np.log(rho/10+3))
	total_cost =  cost + lagrange_mult + penalty 

	return total_cost


def getGradHessBand(loss, B, x):
	# get gradient. create_graph allows higher order derivatives
	delL0, = torch.autograd.grad(loss, x, create_graph=True)
	delL = delL0[:,1:,:].view(B,-1,B) #remove x0
	#y is used to sample the appropriate rows
	#y = torch.zeros(B,N-1,NVars+NControls, requires_grad=False).view(B,-1)
	# There is probably a way to do it this way.
	# Would this be a speed up?
	y = torch.eye(B).view(B,1,B)
	#print(y.shape)
	#print(delL.shape)
	#delL = delL.view(B,-1)
	#y = torch.zeros(B,N-1,NVars+NControls, requires_grad=False).view(B,-1)
	
	#for i in range(B):
	#	y[i,i::B]=1
	#delL = delL.view(B,-1)
	#temp = 0
	#for i in range(B):
	#	temp += torch.sum(delL[i,:,i]) #Direct projection is not faster

	delLy = torch.sum(delL * y)
	delL = delL.view(B,-1)
	
	
	delLy.backward()
	#temp.backward()
	nphess = x.grad[:,1:,:].view(B,-1).detach().numpy()
	#reshuffle columns to actuall be correct
	for i in range(B):
		nphess[:,i::B] = np.roll(nphess[:,i::B], -i+B//2, axis=0)
	#returns gradient and hessian flattened
	return delL.detach().numpy()[0,:].reshape(-1), nphess


def line_search(x, dx, total_cost):
	with torch.no_grad():
		#x1 = torch.unsqueeze(x[0],0)
		xnew = torch.tensor(x) #Make a copy
		alpha = 1
		prev_cost = torch.tensor(total_cost) #Current total cost
		done = False
		# do a backtracking line search
		while not done:
			try:
				xnew[:,1:,:] = x[:,1:,:] - alpha * dx
				#print(xnew.shape)
				total_cost = calc_loss(xnew, l, rho)
				if alpha < 1e-5:
					print("Alpha small: Uh oh")
					done = True
				if total_cost < prev_cost: # - alpha * 0.5 * batch * newton_dec:
					done = True
				else:
					print("Backtrack")
					alpha = alpha * 0.5
			except ValueError: #Sometimes you get NaNs if you have logs in cost func
				print("Out of bounds")
				alpha = alpha * 0.1
		x[:,1:,:] -= alpha * dx #Commit the change
	return x


def opt_iteration(x, l, rho):
	total_cost = calc_loss(x, l, rho)
	gradL, hess = getGradHessBand(total_cost, (NVars+NControls)*3, x)

	#Try to solve the linear system. Sometimes, it fails
	# in which case just defualt to gradient descent
	# you're probably fucked though
	try:
		dx = linalg.solve_banded((bandn,bandn), hess, gradL, overwrite_ab=True)
	except ValueError:
		print("ValueError: Hess Solve Failed.")
		dx = gradL
	except LinAlgError:
		print("LinAlgError: Hess Solve Failed.")
		dx = gradL
	x.grad.data.zero_() # Forgetting this causes awful bugs. I think this has to be here
	newton_dec = np.dot(dx,gradL) # quadratic estimate of cost improvement
	dx = torch.tensor(dx.reshape(1,N-1,NVars+NControls)) # return to original shape
	x = line_search(x, dx, total_cost)

	# If newton decrement is a small percentage of cost, quit
	done = newton_dec < 1e-7*total_cost.detach().numpy()
	return x, done




print("Starting Initial Phase")
#Initial Solve
x, l = getNewState()
rho = 0.0
count = 0
for j in range(6):
	while True:
		count += 1
		print("Count: ", count)
		x, done = opt_iteration(x,l,rho)
		if done:
			break
	with torch.no_grad():
		xres = dynamical_res(x[0].unsqueeze(0))
		print(xres.shape)
		print(l.shape)
		l += 2 * rho * xres
	print("upping rho")
	rho = rho * 10 + 0.1
print("Press Enter for Online Phase")
#input()
#Online Solve
start = time.time()
NT = 1
for t in range(NT): # time steps
	print("Time step")
	with torch.no_grad():
		x[:,0:-1,:] = x[:,1:,:]  # shift forward one step
		l[:,0:-1,:] = l[:,1:,:]
		x[:,0,:] +=  torch.randn(1,NVars+NControls)*0.05
		#x[:,0,:] = x[:,1,:] + torch.randn(1,NVars+NControls)*0.05 #Just move first position
	rho = 100
	for i in range(1): # how many penalty pumping moves
		for m in range(4): # newton steps
			print("Iter Step")
			x, done = opt_iteration(x,l,rho)
		with torch.no_grad():
			xres = dynamical_res(x[0].unsqueeze(0))
			l += 2 * rho * xres
		rho = rho * 10
end = time.time()
print(NT/(end-start), "Hz" )


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
#plt.figure()
#plt.subplot(133)
#plt.plot(costs)
print("hess count: ", count)

plt.show()

