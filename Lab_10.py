#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy
import control
import pandas
import control as con

#Task 1
steps = 100
w = np.arange(1e3,1e6,steps)
R = 1e3
L = 27e-3
C = 100e-9
#%%
mag_H = (w/(R*C)) / np.sqrt(( ((1/(L*C)) - (w**2) )**2) + (w/(R*C))**2)
db_mag_H = 20*np.log10(mag_H)

phase_H = ((.5*np.pi) - np.arctan( (w/(R*C)) / ((1/(L*C)) - (w**2)))) 

for i in range(len(w)):
    if ( ( (1/(L*C)) - w[i]**2 ) < 0 ):
        phase_H[i] -= np.pi

################## Bode for Hand Calculations ##################
myFigSize = (12,8)
plt.figure(figsize=myFigSize)
plt.subplot(1,2,1)
plt.semilogx(w,db_mag_H)
plt.grid(True)
plt.xlabel('w(rad/s)')
plt.ylabel('|H(jw)|(db)')
plt.title('Hand Calculated Magnitude of H(jw)')

plt.subplot(1,2,2)
plt.semilogx(w,phase_H)
plt.grid(True)
plt.xlabel('w (rad/s)')
plt.ylabel('phase angle (rad)')
plt.title('Hand Calculated Phase of H(jw)')
plt.show()
#%%
################## Bode using sig.bode ##################
num = [(1/(R*C)), 0]
den = [1, (1/(R*C)), (1/(L*C))]

(freq, mag, phase) = sig.bode((num, den))

myFigSize = (12,8)
plt.figure(figsize=myFigSize)
plt.subplot(1,2,1)
plt.semilogx(freq, mag)
plt.grid(True)
plt.xlabel('w(rad/s)')
plt.ylabel('|H(jw)|(db)')
plt.title('Magnitude of H(jw) with sig.bode')

plt.subplot(1,2,2)
plt.semilogx(freq, phase)
plt.grid(True)
plt.xlabel('w (rad/s)')
plt.ylabel('phase angle (rad)')
plt.title('Phase of H(jw) with sig.bode')
plt.show()

#%%
################### Frequency plots in Hz###################
sys = con.TransferFunction(num, den)
_= con.bode(sys, w, Hz=True, dB=True, deg=True, Plot=True)
#  _ = ... to suppress the output

#%%
#Task 2
################### Plot a Signal ########################
fs = 1e6
steps = 1/fs
t = np.arange (0, 10e-3 +steps, steps)
x = np.cos(2*np.pi*100*t) + np.cos(2*np.pi*3024*t) + np.sin(2*np.pi*50000*t)

myFigSize = (12,8)
plt.figure(figsize=myFigSize)
plt.subplot(1,1,1)
plt.plot(t, x)
plt.grid(True)
plt.xlabel('t(s)')
plt.ylabel('x(t)')
plt.title('Input signal x(t)')

#%%
################### Pass through RLC circuit ################
fs = 1e9
# z-domain equivalent of x
numX, denX = sig.bilinear(num, den, fs)
#Pass through filter
y = sig.lfilter(numX, denX, x)

plt.figure(figsize=myFigSize)
plt.plot(t, y)
plt.grid(True)
plt.title('Output Signal y(t) through RLC Filter')
plt.xlabel('t(s)')
plt.ylabel('y(t)')

#%%
