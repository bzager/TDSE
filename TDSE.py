# Computational Physics Project 5 - Time Dependent Schrodinger Equation
# 
#

import numpy as np
import scipy as sp
from scipy import linalg
from scipy.sparse import csc_matrix
from scipy.sparse import linalg as splinalg
from scipy.integrate import simps
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons


import time

# constants
L = 100.0 # length of domain [-L/2, L/2]
plotL = L/2
T = 10 # total time

##nx = 1000 # spatial grid size
##nt = 1000 # time grid size
dx = 0.01
dt = 0.01


x = np.arange(-L/2,L/2,dx) # spatial grid
t = np.arange(0,T,dt) # time grid

nx = x.shape[0]
nt = t.shape[0]
##
##dx = x[1] - x[0] # space step size
##dt = t[1] - t[0] # time step size

a = 1/(2*dx**2) # 

def free(x,x0=0,h=0,w=0):
        return 0*x


# harmonic potential
def sho(x,x0=0,h=1,w=0):
        return h*np.square(x-x0)

# potential barrier
# args: position x, left bound of barrier l, right bound of barrier r, height h
# default height is infinite barrier
def wall(x,x0=L/4,h=100,w=0.1):
        U = h*((x>(x0-w/2)) & (x<(x0+w/2)))
        return U

# potential well
# args: position x, left bound of well l, right bound of well r, height h
# default height is infinite well
def well(x,x0=L/4,h=10000,w=0.1):
        U = h*((x<(x0-w/2)) | (x>(x0+w/2)))
        return U

# triangular potential
def triangle(x,x0=0,h=1,w=0):
        return h*np.abs(x)

def Vcomplex(x,x0=0,h=1,w=0):
        U = [h if ((i<x0-w/2) or (i>x0+w/2)) else i*1j/10 for i in x]
        return U

# returns ith diagonal element of Hamiltonian matrix 
def diagonal(i,x):
        return 1 - 2*a + a*(dx**2)*V(x[i])

# constructs Hamiltonian matrix
def constructH(Vs):
        offval = -a
        off = offval*np.ones((nx-1,),dtype=np.complex)
        dia = 2*a+Vs
        dia[0]=dia[-1]=offval
##      dia = np.asarray([diagonal(i,x) for i in range(nx)])
        H = np.diag(dia) + np.diag(off,k=1) + np.diag(off,k=-1)
        
        H[-1,0] = -a # periodic boundary condition
        H[0,-1] = -a

        return H

def constructA(H):
        A = np.eye(nx) + 0.5*1j*dt*H
        return A

def constructB(H):
        B = np.eye(nx) - 0.5*1j*dt*H
        return B

# turns a matrix into a sparse matrix
def sparsify(M):
        return csc_matrix(M)

# initializes psi with initial condition
# and zeros for the rest 
# row n is solution on whole domain at time n
def initPsi(x,x0,A,p0,sig):
        psi = np.zeros([nt,nx],dtype=np.complex)
        psi[0] = packet(x,x0,A,p0,sig)

        return psi

def packet(x,x0,A,p0,sig):
        return A*np.exp(-(x-x0)**2/(2*sig**2) - p0*x*1j)
##      return A*np.exp(p0*(x-x0)*1j)*np.exp(-(x-x0)**2/(4*sig**2))

def BTCS(A,B,psi):
        for n in range(nt-1):
                #psi[n+1] = np..solve(H,psi[n]) # non sparse solver
                psi[n+1] = splinalg.spsolve(A,psi[n]) # sparse solver
        return psi


def CrankNicolson(A,B,psi):
        for n in range(nt-1):
                #psi[n+1] = np.linalg.solve(H,psi[n]) # non sparse solver
                b = B*psi[n]
                b[0]=b[-1]=0
                psi[n+1] = splinalg.spsolve(A,b) # sparse solver
        return psi


# Backward-Time Centered-Space finite difference scheme
# 
def solver(psi, H, method):

        A = constructA(H)
        B = constructB(H)
        A = sparsify(A)
        B = sparsify(B)

        psi = method(A,B,psi)
        
        return psi



def animatePsi(psi, x, areas, Vs, projection='2d', save=False):

        if projection=='2d':
                fig = plt.figure()
                ax = plt.axes(xlim=(-plotL/2,plotL/2),ylim=(-A,A))
                real, = ax.plot([],[],lw=2,color='b',label='Real Part')
                imag, = ax.plot([],[],lw=2,color='r',label='Imaginary Part')
                abso, = ax.plot([],[],lw=2,color='k',label='Absolute Value')
                potent = ax.fill_between(x,np.abs(Vs),color='0.6',label='V(x)', alpha=0.2)
                area = ax.text(0.1,0.9,'',transform=ax.transAxes, ha='left',va='top',fontsize=15)
                plt.legend()

        else:
                fig = plt.figure()
                ax = plt.axes(projection='3d')
        ##      enax=ax.twinx()
        ##      enax = plt.axes(xlim=(-plotL/2,plotL/2),ylim=(-A,A),projection='3d')

                real, = ax.plot([],[],[],lw=2,color='b',label='Real Part')
                imag, = ax.plot([],[],[],lw=2,color='r',label='Imaginary Part')
                abso, = ax.plot([],[],lw=2,color='k',label='Absolute Value')
                comp, = ax.plot([],[],[],lw=2,color='m',label='Complex')
                potent = ax.plot_wireframe(x,np.imag(Vs),np.real(Vs),color='0.6',label='V(x)', alpha=1)
                ax.set_xlim([-plotL/2,plotL/2])
                ax.set_ylim([-A,A])
                ax.set_zlim([-A,A])


                # Create cubic bounding box to simulate equal aspect ratio - Shamelessly stolen from:
                #   http://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
                X = [-plotL/2,plotL/2]
                Y = np.imag(psi[:,0])
                Z = np.real(psi[:,0])
                max_range = np.array([max(X)-min(X), max(Y)-min(Y), max(Z)-min(Z)]).max()
                Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(max(X)+min(X))
                Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(max(Y)+min(Y))
                Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(max(Z)+min(Z))

                for xb, yb, zb in zip(Xb, Yb, Zb):
                   ax.plot([xb], [yb], [zb], 'w')



        def makeAnimation(psi,x,areas, projection, save):
                def animate(i):
                        y = psi[i]
                        if projection=='2d':
                                imag.set_data(x,np.imag(y))
                                real.set_data(x,np.real(y))
                                abso.set_data(x,np.absolute(y))
                                area.set_text("Total Area: "+str(np.round(areas[i],5)))
                                return real,imag,abso,area,potent
                        else:                        
                                real.set_data(x,0)
                                real.set_3d_properties(np.real(y))

                                imag.set_data(x,np.imag(y))
                                imag.set_3d_properties(0)

                                comp.set_data(x,np.imag(y))
                                comp.set_3d_properties(np.real(y))
                                return comp,potent#,real,imag


                def anim_init():
                        if projection=='2d':
                                real.set_data([],[])
                                imag.set_data([],[])
                                abso.set_data([],[])
                                area.set_text('')
                                real.set_data([],[])
                                return real,imag,abso,area,potent
                        else:
                                real.set_3d_properties([])
                                imag.set_data([],[])
                                imag.set_3d_properties([])
                                comp.set_data([],[])
                                comp.set_3d_properties([])

                                return comp,potent#,real,imag

                anim = animation.FuncAnimation(fig,animate,range(len(psi)),init_func=anim_init,interval=0.01,blit=False)
                plt.show()
                if save:
                        anim.save(str("3DTDSE_"+str(V.__name__)+'.mp4'),fps=60)

        makeAnimation(psi, x, areas, projection, save)
        



# finds the eigenvalues and eigenvectors
# takes:   real symmetric square matrix
# returns: list of eigenvalues, list of eigenvectors
def find_eigs(H):
        vals, vects = splinalg.eigs(H)
        return vals, vects

# finds the integral of our solution at a certain timestep in an xrange
# takes:   solution array, timestep, low x val, high x val
# returns: definite integral of solution
def likelihood(psi, timestep, xlo, xhi):
        x_rng = np.linspace(xlo, xhi, nx)
        prob = np.square(np.abs(psi[timestep]))

        return simps(prob, x_rng)

def normalizer(psi):
        tot_integ = likelihood(psi, 0, -L/2, L/2)
        return psi/np.sqrt(tot_integ)





if __name__ == '__main__':
        
        A =  1# amplitude
        k = 2 # wave number
        sig = 1 # width of wave packet
        p0 = -4
        E = p0**2/2
        x0 = 0

        V=sho
        Vargs = [0,1,1] # x0, h, w
        Vs = np.array(V(x, *Vargs),dtype=np.complex)


        H = constructH(Vs)

        vals, vects = find_eigs(sparsify(H))
        E = sorted(vals)[0]
        p0 = np.sqrt(2*E)

        psi = initPsi(x,x0,A,p0,sig)

        

##        fig = plt.figure()
##        ax = plt.axes(xlim=(-L/2,L/2),ylim=(-A,A))
##        real, = ax.plot(x,[],lw=2,color='b',label='Real Part')
##        imag, = ax.plot([],[],lw=2,color='r',label='Imaginary Part')
##        abso, = ax.plot([],[],lw=2,color='k',label='Absolute Value')
##        potent = enax.fill_between(x,Vs,color='0.6',label='V(x)', alpha=0.2)
##        eLevel = enax.hlines(E,-plotL/2,plotL/2,color='0.6',label='E', linestyle='dashed')
##
##        area = ax.text(0.1,0.9,'',transform=ax.transAxes, ha='left',va='top',fontsize=15)
##        fig.legend(*(np.append(ax.get_legend_handles_labels(),enax.get_legend_handles_labels(),axis=1)))




        method = CrankNicolson
##        method = BTCS
        psi = solver(psi, H, method)
        norm_psi = normalizer(psi)
        areas = []
        for i in range(0, nt):
            areas.append(likelihood(norm_psi, i, -L/2, L/2))

##
##        fig = plt.figure(1)
##        plt.plot(x,np.absolute(psi[0]),color='k',label='Absolute Value')
##        plt.plot(x,np.real(psi[0]),color='b',label='Real Part')
##        plt.plot(x,np.imag(psi[0]),color='r',label='Imaginary Part')
##        plt.plot(x,U,color='0.8',label='Potential')
##        plt.xlim([-L/2,L/2])
##        plt.ylim(-A,A)
##        plt.title("Initial Condition")
##        plt.legend()
        animatePsi(psi, x, areas, Vs, projection='2d', save=True)
##        anim = animation.FuncAnimation(fig,animate,frames=psi,interval=1, blit=True)

##        plt.show()

