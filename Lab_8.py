#%%
#  -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 19:39:08 2019

@author: ajoli
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import control
import pandas

steps = .1
t = np.arange(0,20+steps,steps)
T = 8
N = [1, 3, 15, 50, 150, 1500]

def b(k):
    bk = np.zeros(k+1)
    for k in range (k+1):
        if (k > 0):
            bk[k] = (2/(k*np.pi))*(1-np.cos(np.pi*k))
        else:
            bk[k] = 0
    return bk

bn = b(1500)

print('b_0: ', bn[0])
print('b_1: ', bn[1])
print('b_2: ', bn[2])
print('b_3: ', bn[3])


myFigSize = (12,8)

#%%
y = np.zeros(len(t))
plt.figure(figsize=myFigSize)
count = 0
for k in range(max(N)+1):
    for t_val in range(len(t)):
        x = bn[k]*np.sin((k*2*np.pi*t[t_val])/T)
        y[t_val] += x

    if k in N:  
        if count == 3:
            plt.figure(figsize=myFigSize)
        plt.subplot(3,1, count%3+1)
        count = count + 1 #indexing variable
        if count == 1 or count == 4:
            plt.title('x(t)')   
        plt.ylabel('N = ' +str(k))
        plt.xlabel('t[s]')
        plt.plot(t,y)
        plt.grid()         
plt.show()

#%%
