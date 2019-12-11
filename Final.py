#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy
import control
import control as con
import pandas as pd
#%%

fs = 1e6
Ts = 1/fs
t_end = 50e-3
t = np.arange (0,t_end-Ts,Ts)

# load input signal
df = pd.read_csv ('NoisySignal.csv')

t = df ['0'].values
sensor_sig = df ['1'].values
plt.figure(figsize = (10 , 7) )
plt.plot (t, sensor_sig )
plt.grid ()
plt.title ('Noisy Input Signal')
plt.xlabel ('Time [s]')
plt.ylabel ('Amplitude [V]')
plt.show ()

#%%
# User defined clean fft function 
def clean_fft(x,fs):
    
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
# Work- Around
def make_stem ( ax ,x ,y , color ='k', style ='solid', label ='', linewidths =2.5 ,** kwargs) :
    ax.axhline ( x [0] , x [ -1] ,0 , color ='r')
    ax.vlines (x , 0 ,y , color = color , linestyles = style , label = label , linewidths = linewidths )
    ax.set_ylim ([1.05* y . min () , 1.05* y . max () ])
#%%
# FFT plots
(freq, X_mag, X_phi) = clean_fft(sensor_sig,fs) 

# All frequencies
fig, ax1 = plt.subplots( figsize =(10,7) )
make_stem(ax1,freq,X_mag)
plt.xlim(-10000, 500e3)
plt.grid(True)
plt.xlabel('w (Hz)')
plt.ylabel('H(jw) dB')
plt.title('Total_signal fft')
plt.show()
#%%
# Low frequencies
fig, ax1 = plt.subplots( figsize =(10,7) )
make_stem(ax1,freq,X_mag)
plt.grid(True)
plt.xlim(1e0, 1800)
plt.xlabel('w (Hz)')
plt.ylabel('H(jw) dB')
plt.title('Total_signal Low Frequencies')
plt.show()

#%%
# High frequencies
fig, ax1 = plt.subplots( figsize =(10,7) )
make_stem(ax1,freq,X_mag)
plt.grid(True)
plt.xlim(2000, 500e3)
plt.xlabel('w (Hz)')
plt.ylabel('H(jw) dB')
plt.title('Total_signal High Frequencies')
plt.show()


#%%
# Middle frequencies
fig, ax1 = plt.subplots( figsize =(10,7) )
make_stem(ax1,freq,X_mag)
plt.grid(True)
plt.xlim(1800, 2000)
plt.xlabel('w (Hz)')
plt.ylabel('H(jw) dB')
plt.title('Total_signal Middle Frequencies')
plt.show()
#%%
# Transfer Function for filter 
steps = 1e0
w = np.arange(1e0, fs ,steps)
R = 4122
L = 200e-3
C =35.1e-9

num = [(1/(R*C)), 0]
den = [1, (1/(R*C)), (1/(L*C))]

mag_H = (w/(R*C)) / np.sqrt(( ((1/(L*C)) - (w**2) )**2) + (w/(R*C))**2)
db_mag_H = 20*np.log10(mag_H)

phase_H = ((.5*np.pi) - np.arctan( (w/(R*C)) / ((1/(L*C)) - (w**2)))) 

for i in range(len(w)):
    if ( ( (1/(L*C)) - w[i]**2 ) < 0 ):
        phase_H[i] -= np.pi

# Bode Plots

# All frequencies
plt.figure(figsize=(10,7))
plt.title('Bode Plot at all Frequencies')
sys = con.TransferFunction(num, den)
_= con.bode(sys, w, Hz=True, dB=True, deg=True, Plot=True)
plt.show()

# Low Frequencies
w = np.arange(1e0, 1800+steps,steps)*2*np.pi
plt.figure(figsize=(10,7))
plt.title('Bode Plot at Low Frequencies')
sys = con.TransferFunction(num, den)
_= con.bode(sys, w, Hz=True, dB=True, deg=True, Plot=True)
plt.show()

# Middle Frequencies
w = np.arange(1800, 2000+steps,steps)*2*np.pi
plt.figure(figsize=(10,7))
plt.title('Bode Plot at Middle Frequencies')
sys = con.TransferFunction(num, den)
_= con.bode(sys, w, Hz=True, dB=True, deg=True, Plot=True)
plt.show()

# High Frequencies
w = np.arange(2000, fs+steps,steps)*2*np.pi
plt.figure(figsize=(10,7))
plt.title('Bode Plot at High Frequencies')
sys = con.TransferFunction(num, den)
_= con.bode(sys, w, Hz=True, dB=True, deg=True, Plot=True)
plt.show()

# Pass through Bandpass RLC circuit 

# z-domain equivalent of x
numX, denX = sig.bilinear(num, den, fs)
#Pass through filter
y = sig.lfilter(numX, denX, sensor_sig)

myFigSize = (12,8)
plt.figure(figsize=myFigSize)
plt.plot(t, y)
plt.grid(True)
plt.title('Output Signal y(t) through RLC Filter')
plt.xlabel('t(s)')
plt.ylabel('y(t)')

# %%
# FFT of filtered signal
clean_sig = y
(freq, X_mag, X_phi) = clean_fft(clean_sig,fs) 

# All frequencies
fig, ax1 = plt.subplots( figsize =(10,7) )
make_stem(ax1,freq,X_mag)
plt.xscale('log')
plt.xlim(1e0, fs)
plt.xlabel('w (Hz)')
plt.ylabel('H(jw) dB')
plt.title('Total_signal fft')
plt.show()

# %%
