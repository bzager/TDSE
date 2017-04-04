# Computational Physics Project 5 - Time Dependent Schrodinger Equation
# 
#

import numpy as np
import scipy as sp
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib import animation

# constants
L = 5.0 # length of domain [-L/2, L/2]
T = 5.0 # total time 
nx = 100 # spatial grid size
nt = 100 # time grid size

x = np.linspace(-L/2,L/2,nx) # spatial grid

dx = L / nx # space step size
dt = T / nt # time step size

a = (dt/dx**2) * 1.0j # 

# potential function
def V(x):
	#U = 0 	# free particle
	#U = harmonic(x)
	#U = well(x)
	U = triangle(x)
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
def constructH(x,nx,nt,a):
	off = a*np.ones((nx-1,),dtype=complex)
	dia = np.asarray([diagonal(i,x) for i in range(nx)])
	H = np.diag(dia) + np.diag(off,k=1) + np.diag(off,k=-1)    
	return H

# initializes psi with initial condition
# and zeros for the rest 
# row n is solution on whole domain at time n
def initPsi(nx,nt,x,k,sig):
	psi = np.zeros([nt,nx],dtype=complex)
	psi[0,:] = packet(x,k,sig)
	return psi

def packet(x,k,sig):
	return np.exp(-x**2/(4*sig**2) + k*x*1.0j)

def BTCS(x,nx,nt,a,k,sig):

	H = constructH(x,nx,nt,a)
	psi = initPsi(nx,nt,x,k,sig)

	for n in range(nt-1):
		psi[n+1,:] = np.linalg.solve(H,psi[n,:])
		psi[n+1,0] = 0
		psi[n+1,-1] = 0 

	return psi

k = 1
sig = 0.3

psi = BTCS(x,nx,nt,a,k,sig)
U = np.asarray([V(x[i]) for i in range(nx)])

plt.plot(x,np.absolute(psi[0,:]),color='k',label='Absolute Value')
plt.plot(x,np.real(psi[0,:]),color='b',label='Real Part')
plt.plot(x,np.imag(psi[0,:]),color='r',label='Imaginary Part')
plt.plot(x,U,color='0.8',label='Potential')
plt.xlim([-L/2,L/2])
plt.title("Initial Condition")
plt.legend()
plt.show()

"""      
time = 80
plt.plot(x,np.absolute(psi[time,:]),color='k',label='Absolute Value')
plt.plot(x,np.real(psi[time,:]),color='b',label='Real Part')
plt.plot(x,np.imag(psi[time,:]),color='r',label='Imaginary Part')
plt.xlim([-L/2,L/2])
plt.title("Solution at t="+str(T*time/nt))
plt.show()
"""

fig = plt.figure()
ax = plt.axes(xlim=(-L/2,L/2),ylim=(-1,1))
real, = ax.plot([],[],lw=2,color='b')
imag, = ax.plot([],[],lw=2,color='r')
abso, = ax.plot([],[],lw=2,color='k')
potent = ax.plot(x,U,color='0.6')

anim = animation.FuncAnimation(fig,animate,frames=psi,interval=100)
plt.show()

