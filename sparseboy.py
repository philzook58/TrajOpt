import sparsegrad.forward as forward
import numpy as np
import osqp
import scipy.sparse as sparse
import matplotlib.pyplot as plt

#def f(x):
#	return x**2

#ad.test()
#print(dir(ad))
N = 100
NVars  = 3
#Px = sparse.eye(N)
#sparse.csc_matrix((N, N)) 
# The three deifferent weigthing matrices for x, v, and external force
P = sparse.block_diag([sparse.diags(np.arange(N)/N),sparse.eye(N)*0.01 ,sparse.eye(N)*0.1])
#P[N,N]=10
q = np.zeros((NVars, N))
q[0,:] = -0.5*np.arange(N)/N
#q[N,0] = -2 * 0.5 * 10
q = q.flatten()
#u = np.arr

x = np.zeros((N,NVars)).flatten()
#v = np.zeros(N)
#f = np.zeros(N)


#print(f(ad.seed(x)).dvalue)




def constraint(var, x0, v0):
	#x[0] -= 1
	#print(x[0])
	x = var[:N]
	v = var[N:2*N]
	a = var[2*N:3*N]

	avgx = (x[0:-1]+x[1:])/2
	avgv = (v[0:-1]+v[1:])/2
	dt1 = 1/0.1
	dx = (x[0:-1]-x[1:]) * dt1
	dv = (v[0:-1]-v[1:]) * dt1
	f = -np.sin(avgx) + a[1:] #return v - f()
	vres = dv - f
	xres = dx - avgv
	return x[0:1]-x0, v[0:1]-v0, xres,vres 
	#return x[0:5] - 0.5


cons = constraint(forward.seed_sparse_gradient(x), 0.1, 0)
A = sparse.vstack(map(lambda z: z.dvalue, cons)).tocsc() #  y.dvalue.tocsc()
#print(tuple(map(lambda z: z.value, cons)))
totval = np.concatenate(tuple(map(lambda z: z.value, cons)))
temp = A@x - totval
u = temp
l = temp
#A = y.dvalue.tocsc()
#print(y.dvalue)
#sparse.hstack( , format="csc")

m = osqp.OSQP()
m.setup(P=P, q=q, A=A, l=l, u=u) #  **settings
results = m.solve()
print(results.x)

plt.plot(results.x[:N])
plt.plot(results.x[N:2*N])
plt.plot(results.x[2*N:3*N])

#m.update(Px=Px_new, Px_idx=Px_new_idx, Ax=Ax_new, Ax=Ax_new_idx)

plt.show()
#m.update(q=q_new, l=l_new, u=u_new)
