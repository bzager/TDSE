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
T = 0.005 # total time

nx = 400 # spatial grid size
nt = 800 # time grid size

x = np.linspace(-L/2,L/2,nx) # spatial grid
t = np.linspace(0,T,nt) # time grid

dx = x[1] - x[0] # space step size
dt = t[1] - t[0] # time step size

a = 1/dx**2 # 

# potential function
def V(x):
	#U = 0 	# free particle
	#U = harmonic(x)
	#U = well(x)
	#U = triangle(x)
	#U = step(x)
	U = barrier(x)
	return U

# harmonic potential
def harmonic(x,h=100000):
	return h*np.square(x)

# potential well
# args: position x, left bound of well l, right bound of well r, height h
# default height is infinite well
def well(x,l=-0.1*L,r=0.1*L,h=10000000):
	if x < l or x > r:
		U = h
	else:
		U = 0
	return U

# triangular potential
def triangle(x):
	return np.abs(x)

# potential step
def step(x,pos=0.3*L,h=10000000):
	if x > pos:
		return h
	else:
		return 0

# potential barrier
def barrier(x,l=0.1*L,r=0.2*L,h=10000000):
	if x > l and x < r:
		return h
	else:
		return 0

# returns ith diagonal element of Hamiltonian matrix 
def diagonal(i,x):
	return 1 - 2*a + a*(dx**2)*V(x[i])

# constructs Hamiltonian matrix
def constructH():
	off = a*np.ones((nx-1,),dtype=complex)
	dia = np.asarray([diagonal(i,x) for i in range(nx)])
	H = np.diag(dia) + np.diag(off,k=1) + np.diag(off,k=-1)
	
	H[-1,0] = a # periodic boundary condition
	H[0,-1] = a

	return H

def constructA(H):
	A = np.eye(nx) + 1j*dt*H
	return A

def constructB(H):
	B = np.eye(nx) - 1j*dt*H
	return A

# turns a matrix into a sparse matrix
def sparsify(A):
	return csc_matrix(A)

# initializes psi with initial condition
# and zeros for the rest 
# row n is solution on whole domain at time n
def initPsi(x0,A,k,sig):
	psi = np.zeros([nt,nx],dtype=complex)
	psi[0,:] = packet(x0,A,k,sig)

	return psi

def packet(x0,A,k,sig):
	return A*np.exp(-(x-x0)**2/(2*sig**2) - k*x*1j)

def BTCS(A,B,psi):

	for n in range(nt-1):
		#psi[n+1,:] = np.linalg.solve(H,psi[n,:]) # non sparse solver
		psi[n+1,:] = linalg.spsolve(A,psi[n,:]) # sparse solver

	return psi


def CrankNicolson(A,B,psi):
	for n in range(nt-1):
		#psi[n+1,:] = np.linalg.solve(H,psi[n,:]) # non sparse solver
		b = B*psi[n,:]
		psi[n+1,:] = linalg.spsolve(A,b) # sparse solver

	return psi


# Backward-Time Centered-Space finite difference scheme
# 
def solver(nx,nt,x,x0,A,k,sig,method):

	H = constructH()
	A = constructA(H)
	B = constructB(H)
	A = sparsify(A)
	psi = initPsi(x0,A,k,sig)

	psi = method(A,B,psi)
	
	return psi

x0 = -0.2
A = 1 # amplitude
k = 100 # wave number
sig = 0.1 # width of wave packet

times = []

method = CrankNicolson
#method = BTCS

psi = solver(nx,nt,x,x0,A,k,sig,method)
U = np.asarray([V(x[i]) for i in range(nx)])


# Animate solution
def animHelper(data):
    y = data
    real.set_data(x,np.real(y))
    imag.set_data(x,np.imag(y))
    abso.set_data(x,np.absolute(y))
    return real,imag,abso

def animate(psi):
	fig = figure()
	ax = plt.axes(xlim=(-L/2,L/2),ylim=(-A,A))
	real, = ax.plot([],[],lw=2,color='b',label='Real Part')
	imag, = ax.plot([],[],lw=2,color='r',label='Imaginary Part')
	abso, = ax.plot([],[],lw=2,color='k',label='Absolute Value')
	potent = ax.plot(x,U,color='0.6',label='V(x)')
	plt.legend()

	anim = animation.FuncAnimation(fig,animHelper,frames=psi,interval=1)

	return anim

fig = plt.figure(1)
plt.plot(x,np.absolute(psi[0,:]),color='k',label='Absolute Value')
plt.plot(x,np.real(psi[0,:]),color='b',label='Real Part')
plt.plot(x,np.imag(psi[0,:]),color='r',label='Imaginary Part')
plt.plot(x,U,color='0.8',label='Potential')
plt.xlim([-L/2,L/2])
plt.ylim(-A,A)
plt.title("Initial Condition")
plt.legend()


anim = animation.FuncAnimation(fig,anim,frames=psi,interval=1)

#times.append(time.time())
#timeDif = np.diff(times)
#print(timeDif)

plt.show()

