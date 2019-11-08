# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:53:35 2019

@author: ajoli
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy
import control
import pandas
#%%

fs = 100
steps = 1/fs
t = np.arange(0,2,steps)
x = 0
myFigSize = (12,8)


def func1(t): 
    a = np.zeros((len(t),1))
    for i in range(len(t)):
        a[i] = np.cos(2*np.pi*t[i])
    return a

a = func1(t)

def func2(t): 
    b = np.zeros((len(t), 1))
    for i in range(len(t)):
        b[i] = 5*np.sin(2*np.pi*t[i])
    return b 
 
b = func2(t)

def func3(t): 
    c = np.zeros((len(t), 1))
    for i in range(len(t)):
        c[i] = 2*np.cos((4*np.pi*t[i])-2)+np.sin((12*np.pi*t[i])+3)**2
    return c 
  
c = func3(t)
#
##Userdefined fft function
# 
def fft(x,fs):
    
    N = len( x ) # find the length of the signal
    X_fft = scipy.fftpack.fft ( x ) # perform the fast Fourier transform (fft)
    X_fft_shifted = scipy.fftpack.fftshift ( X_fft )    # shift zero frequency components
                                                            # to the center of the spectrum

    freq = np.arange ( - N /2 , N /2) * fs / N        # compute the frequencies for the output
                                                        # signal , (fs is the sampling frequency and
                                                        # needs to be defined previously in your code
    X_mag = np.abs( X_fft_shifted ) / N # compute the magnitudes of the signal
    X_phi = np.angle ( X_fft_shifted ) # compute the phases of the signal

    return (freq, X_mag, X_phi)


#%% Task 1
x=np.cos(2*np.pi*t)
(freq, X_mag, X_phi) = fft(x,100) 

plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t,a)
plt.grid(True)
plt.xlabel('t(s)')
plt.ylabel('x(t)')
plt.title('Task 1 -User-Defined FFT of x(t)')

plt.subplot(3,2,3)
plt.stem ( freq , X_mag, use_line_collection= True ) # you will need to use stem to get these plots to be
plt.grid(True)
plt.ylabel('|x(f)|')

plt.subplot(3,2,4)
plt.xlim(-2, 2)
plt.stem ( freq , X_mag, use_line_collection= True ) # you will need to use stem to get these plots to be
plt.grid(True)

plt.subplot(3,2,5)
plt.stem ( freq , X_phi, use_line_collection= True ) # correct , remember to label all plots appropriately
plt.grid(True)
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

plt.subplot(3,2,6)
plt.xlim(-2, 2)
plt.stem ( freq , X_phi, use_line_collection= True ) # correct , remember to label all plots appropriately
plt.grid(True)
plt.xlabel('f[Hz]')
#%%
##Task 2
x = 5*np.sin(2*np.pi*t)
(freq, X_mag, X_phi) = fft(x,100) 

plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t,b)
plt.grid(True)
plt.xlabel('t(s)')
plt.ylabel('x(t)')
plt.title('Task 2 -User-Defined FFT of x(t)')

plt.subplot(3,2,3)
plt.stem ( freq , X_mag, use_line_collection= True ) # you will need to use stem to get these plots to be
plt.grid(True)
plt.ylabel('|x(f)|')

plt.subplot(3,2,4)
plt.xlim(-2, 2)
plt.stem ( freq , X_mag, use_line_collection= True ) # you will need to use stem to get these plots to be
plt.grid(True)

plt.subplot(3,2,5)
plt.stem ( freq , X_phi, use_line_collection= True ) # correct , remember to label all plots appropriately
plt.grid(True)
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

plt.subplot(3,2,6)
plt.xlim(-2, 2)
plt.stem ( freq , X_phi, use_line_collection= True ) # correct , remember to label all plots appropriately
plt.grid(True)
plt.xlabel('f[Hz]')
plt.show()
#%%
##Task 3
x = 2*np.cos((4*np.pi*t)-2)+np.sin((12*np.pi*t)+3)**2
(freq, X_mag, X_phi) = fft(x,100) 

plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t,c)
plt.grid(True)
plt.xlabel('t(s)')
plt.ylabel('x(t)')
plt.title('Task 3 -User-Defined FFT of x(t)')

plt.subplot(3,2,3)
plt.stem ( freq , X_mag, use_line_collection= True ) # you will need to use stem to get these plots to be
plt.grid(True)
plt.ylabel('|x(f)|')

plt.subplot(3,2,4)
plt.xlim(-15, 15)
plt.stem ( freq , X_mag, use_line_collection= True ) # you will need to use stem to get these plots to be
plt.grid(True)


plt.subplot(3,2,5)
plt.stem ( freq , X_phi, use_line_collection= True ) # correct , remember to label all plots appropriately
plt.grid(True)
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

plt.subplot(3,2,6)
plt.xlim(-15, 15)
plt.stem ( freq , X_phi, use_line_collection= True ) # correct , remember to label all plots appropriately
plt.grid(True)
plt.xlabel('f[Hz]')
plt.show()


#%% Task 4 
def fft(x,fs):
    
    N = len( x ) # find the length of the signal
    X_fft = scipy.fftpack.fft ( x ) # perform the fast Fourier transform (fft)
    X_fft_shifted = scipy.fftpack.fftshift ( X_fft )    # shift zero frequency components
                                                            # to the center of the spectrum

    freq = np.arange ( - N /2 , N /2) * fs / N        # compute the frequencies for the output
                                                        # signal , (fs is the sampling frequency and
                                                        # needs to be defined previously in your code
    X_mag = np.abs( X_fft_shifted ) / N # compute the magnitudes of the signal
    X_phi = np.angle ( X_fft_shifted ) # compute the phases of the signal
    for i in range(len(X_mag)):
        if np.abs(X_mag[i]) < 1e-10:
            X_phi[i] = 0
    return (freq, X_mag, X_phi)

#%%
#Task 4: func 1
x=np.cos(2*np.pi*t)
(freq, X_mag, X_phi) = fft(x,100) 

plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t,a)
plt.grid(True)
plt.xlabel('t(s)')
plt.ylabel('x(t)')
plt.title('Task 4 x=cos(2*pi*t) clean fft')

plt.subplot(3,2,3)
plt.stem ( freq , X_mag, use_line_collection= True ) # you will need to use stem to get these plots to be
plt.grid(True)
plt.ylabel('|x(f)|')

plt.subplot(3,2,4)
plt.xlim(-2, 2)
plt.stem ( freq , X_mag, use_line_collection= True ) # you will need to use stem to get these plots to be
plt.grid(True)

plt.subplot(3,2,5)
plt.stem ( freq , X_phi, use_line_collection= True ) # correct , remember to label all plots appropriately
plt.grid(True)
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

plt.subplot(3,2,6)
plt.xlim(-2, 2)
plt.stem ( freq , X_phi, use_line_collection= True ) # correct , remember to label all plots appropriately
plt.grid(True)
plt.xlabel('f[Hz]')
plt.show()
#%%
#Task 4: func 2
x=np.cos(2*np.pi*t)
(freq, X_mag, X_phi) = fft(x,100) 

plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t,b)
plt.grid(True)
plt.xlabel('t(s)')
plt.ylabel('x(t)')
plt.title('Task 4 x=5sin(2*pi*t) clean fft')

plt.subplot(3,2,3)
plt.stem ( freq , X_mag, use_line_collection= True ) # you will need to use stem to get these plots to be
plt.grid(True)
plt.ylabel('|x(f)|')

plt.subplot(3,2,4)
plt.xlim(-2, 2)
plt.stem ( freq , X_mag, use_line_collection= True ) # you will need to use stem to get these plots to be
plt.grid(True)

plt.subplot(3,2,5)
plt.stem ( freq , X_phi, use_line_collection= True ) # correct , remember to label all plots appropriately
plt.grid(True)
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

plt.subplot(3,2,6)
plt.xlim(-2, 2)
plt.stem ( freq , X_phi, use_line_collection= True ) # correct , remember to label all plots appropriately
plt.grid(True)
plt.xlabel('f[Hz]')
plt.show()
#%%
#Task 4: func 3
x=np.cos(2*np.pi*t)
(freq, X_mag, X_phi) = fft(x,100) 

plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t,c)
plt.grid(True)
plt.xlabel('time')
plt.ylabel('x(t)')
plt.title('Task 4 x= 2cos((4*pi*t)-2)+sin((12*pi*t)+3)**2 clean fft')

plt.subplot(3,2,3)
plt.stem ( freq , X_mag, use_line_collection= True ) # you will need to use stem to get these plots to be
plt.grid(True)
plt.ylabel('|x(f)|')

plt.subplot(3,2,4)
plt.xlim(-15, 15)
plt.stem ( freq , X_mag, use_line_collection= True ) # you will need to use stem to get these plots to be
plt.grid(True)


plt.subplot(3,2,5)
plt.stem ( freq , X_phi, use_line_collection= True ) # correct , remember to label all plots appropriately
plt.grid(True)
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

plt.subplot(3,2,6)
plt.xlim(-15, 15)
plt.stem ( freq , X_phi, use_line_collection= True ) # correct , remember to label all plots appropriately
plt.grid(True)
plt.xlabel('f[Hz]')
plt.show()

#%%
#Task 5

fs = 100
steps = 1/fs
t = np.arange(0,2,steps)
myFigSize = (12,8)


k = np.arange (0,16)
bk = (2/(k*np.pi))*(1-np.cos(np.pi*k))
    
T = 8
t = np.arange(0,16,steps)

x = np.zeros(len(t))
for k in np.arange(1,15):
    x += bk[k]*np.sin((k*2*np.pi*t)/T)

(freq, X_mag, X_phi) = fft(x,100) 

plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t,x)
plt.grid(True)
plt.xlabel('time')
plt.ylabel('x(t)')
plt.title('Square wave using clean fft')

plt.subplot(3,2,3)
plt.stem ( freq , X_mag, use_line_collection= True ) # you will need to use stem to get these plots to be
plt.grid(True)
plt.ylabel('|x(f)|')

plt.subplot(3,2,4)
plt.xlim(-15, 15)
plt.stem ( freq , X_mag, use_line_collection= True ) # you will need to use stem to get these plots to be
plt.grid(True)

plt.subplot(3,2,5)
plt.stem ( freq , X_phi, use_line_collection= True ) # correct , remember to label all plots appropriately
plt.grid(True)
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

plt.subplot(3,2,6)
plt.xlim(-15, 15)
plt.stem ( freq , X_phi, use_line_collection= True ) # correct , remember to label all plots appropriately
plt.grid(True)
plt.xlabel('f[Hz]')
plt.show()

#%%