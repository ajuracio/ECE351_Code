# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import control
import pandas

#Part 1: Plot y(t)
steps = 1e-3
t = np.arange(0,2+steps,steps)

#Hand calculation 

#Defined unit function
def u(t):
    y = np.zeros(t.shape)
    for i in range (len(t)):
        if t[i] >= 0:
            y[i]=1
                
        else:
            y[i]=0
            
    return y 
    
    
#Defined y(t) functions   
y = (1/2+np.exp(-6*t)-(1/2)*np.exp(-4*t))*u(t)
    
#Python calculation
num = [1,6,12]
den = [1,10,24]
tout , yout = sig.step(( num , den ) , T = t )


#Graph for hand calc h(t) 
myFigSize = (10,8)
plt.figure(figsize=myFigSize)
plt.subplot(1,2,1)
plt.plot(t,y)
plt.grid(True) 
plt.xlabel('time')
plt.ylabel('y(t)')
plt.title('Hand calculated y(t)')

#Graph for python h(t) 
plt.subplot(1,2,2)
plt.plot(tout,yout)
plt.grid(True)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.title('Python calculated y(t)')
plt.show()

#Python partial fraction expantion 
num = [1,6,12]
den = [1,10,24, 0]
[R,P,_]=sig.residue( num, den )

print('R1: ', R, '\nP1: ', P)


#Part 2:  Find zeros and poles

#Python zeros and poles
den = [1, 18, 218, 2036, 9085, 25250, 0]
num = [25250]
[R,P,_]=sig.residue( num, den )

print('R2: ',R, '\nP2: ',P)

#Cosine method 
t = np.arange(0,4.5+steps,steps)
y = np.zeros(t.shape)

for i in range (len(R)):  
    k = abs(R[i])
    a = np.real(P[i])
    w = np.imag(P[i])
    ang = np.angle(R[i])
    
    y = y + (k*np.exp(a*t)*np.cos((w*t)+ang))*u(t)

 

#sig.step method
den = [1, 18, 218, 2036, 9085, 25250]
num = [25250]
tout , yout = sig.step(( num , den ) , T = t )



#Graphed cosine method
plt.figure(figsize=myFigSize)
plt.subplot(1,2,1)
plt.plot(t,y)
plt.grid(True)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.title('Cosine method')

#Graphed h(t)*u(t)

plt.subplot(1,2,2)
plt.plot(tout,yout)
plt.grid(True)
plt.xlabel('time')
plt.ylabel('h(t)*u(t)')
plt.title('Using sig.step')
#plt.show()












