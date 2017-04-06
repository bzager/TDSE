# Computational Physics Project 5 - Time Dependent Schrodinger Equation
# 
#

import numpy as np
import scipy as sp
from scipy import linalg
from scipy.sparse import csc_matrix,linalg
import matplotlib.pyplot as plt
from matplotlib import animation

import time

# constants
L = 1.0 # length of domain [-L/2, L/2]
T = 1.0 # total time

nx = 200 # spatial grid size
nt = 100 # time grid size

x = np.linspace(-L/2,L/2,nx) # spatial gri
t = np.linspace(0,T,nt) # time grid

dx = x[1] - x[0] # space step size
dt = t[1] - t[0] # time step size

a = (dt/dx**2)*1.0j # 

# potential function
def V(x):
	#U = 0 	# free particle
	#U = harmonic(x)
	U = well(x)
	#U = triangle(x)
	return U

# harmonic potential
def harmonic(x):
	return np.square(x)

# potential well
# args: position x, left bound of well l, right bound of well r, height h
# default height is infinite well
def well(x,l=-0.2*L,r=0.2*L,h=10000):
	if x < l or x > r:
		U = h
	else:
		U = 0
	return U

# triangular potential
def triangle(x):
	return np.abs(x)

# Animate solution
def animate(data):
    x = np.linspace(-L/2,L/2,nx)
    y = data
    real.set_data(x,np.real(y))
    imag.set_data(x,np.imag(y))
    abso.set_data(x,np.absolute(y))
    return real,imag,abso

# returns ith diagonal element of Hamiltonian matrix 
def diagonal(i,x):
	return 1 - 2*a + a*(dx**2)*V(x[i])

# constructs Hamiltonian matrix
def constructH(x,nx,nt):
	off = a*np.ones((nx-1,),dtype=complex)
	dia = np.asarray([diagonal(i,x) for i in range(nx)])
	H = np.diag(dia) + np.diag(off,k=1) + np.diag(off,k=-1)
	
	H[-1,0] = a # periodic boundary condition
	H[0,-1] = a

	return H

# turns a matrix into a sparse matrix
def sparsify(A):
	return csc_matrix(A)

# initializes psi with initial condition
# and zeros for the rest 
# row n is solution on whole domain at time n
def initPsi(nx,nt,x,A,k,sig):
	psi = np.zeros([nt,nx],dtype=complex)
	psi[0,:] = packet(x,A,k,sig)

	return psi

def packet(x,A,k,sig):
	return A*np.exp(-(x)**2/(4*sig**2) + k*x*1.0j)

# Backward-Time Centered-Space finite difference scheme
def BTCS(x,nx,nt,A,k,sig):

	H = constructH(x,nx,nt)
	H = sparsify(H)
	psi = initPsi(nx,nt,x,A,k,sig)

	#times.append(time.time())

	for n in range(nt-1):
		#psi[n+1,:] = np.linalg.solve(H,psi[n,:]) # non sparse solver
		psi[n+1,:] = linalg.spsolve(H,psi[n,:]) # sparse solver
		psi[n+1,-1] = 0.5*(psi[n+1,1] + psi[n+1,-2])

	#times.append(time.time())

	return psi

A = 0.5 # amplitude
k = 2 # wave number
sig = 0.02 # width of wave packet

times = []
psi = BTCS(x,nx,nt,A,k,sig)

U = np.asarray([V(x[i]) for i in range(nx)])

fig = plt.figure(1)
plt.plot(x,np.absolute(psi[0,:]),color='k',label='Absolute Value')
plt.plot(x,np.real(psi[0,:]),color='b',label='Real Part')
plt.plot(x,np.imag(psi[0,:]),color='r',label='Imaginary Part')
plt.plot(x,U,color='0.8',label='Potential')
plt.xlim([-L/2,L/2])
plt.ylim(-1,1)
plt.title("Initial Condition")
plt.legend()

fig = plt.figure(2)
ax = plt.axes(xlim=(-L/2,L/2),ylim=(-1,1))
real, = ax.plot([],[],lw=2,color='b',label='Real Part')
imag, = ax.plot([],[],lw=2,color='r',label='Imaginary Part')
abso, = ax.plot([],[],lw=2,color='k',label='Absolute Value')
potent = ax.plot(x,U,color='0.6',label='V(x)')
plt.legend()

anim = animation.FuncAnimation(fig,animate,frames=psi,interval=100)

#times.append(time.time())
#timeDif = np.diff(times)
#print(timeDif)

plt.show()

