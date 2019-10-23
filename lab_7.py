# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 19:36:40 2019

@author: ajoli
"""

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

#Part 1: Block diagram analysis - open loop
    #1:G(s), A(s), and B(s) in factored form

    #G(s)
numG = [1,9]
denG = [1,-2,-40,-64]
[R,P,K] = sig.tf2zpk( numG, denG)
print('R1: ', R, '\nP1: ', P, '\nK1: ', K)

    #A(s)
numA = [1,4]
denA = [1,4,3]
[R,P,K] = sig.tf2zpk( numA, denA)
print('R2: ', R, '\nP2: ', P, '\nK2: ', K)

    #B(s)
numB = [1,26,168]
denB = [1]
[R,P,K] = sig.tf2zpk( numB, denB)
print('R3: ', R, '\nP3: ', P, '\nK3: ', K)

    #2: scipy.signal.tf2zpk() function to check your results
    #open loop
    
num = sig.convolve(numA, numG)
den = sig.convolve(denA, denG)
[R,P,K] = sig.tf2zpk( num, den)
print('R_OPEN: ', R, '\nP_OPEN: ', P, '\nK_OPEN: ', K)

    #3:Type and properly format the open-loop transfer function where x(t) is the input and y(t) is
    #  the output. Keep your answer in factored form.

    #4:Considering the expression found in Task 3, is the open-loop response stable? Explain why
    #  or why not.

    #No, we have a positive zero value
    
    #5:Plot the step response of the open-loop transfer function.
     #Graph for python h(t) 
steps = 1e-3
t = np.arange(0,4.5+steps,steps)   
tout , yout = sig.step(( num , den ) , T = t )
 
myFigSize = (15,10)
plt.subplot(1,2,1)
plt.plot(tout,yout)
plt.grid(True)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.title('Open loop y(t)')
 

    #6:Does your result from Task 5 support your answer from Task 4?
    
#Part 2: Block diagram analysis - closed loop
    
numCL = [sig.convolve(numA, numG)]
denCL = sig.convolve((denG + sig.convolve(numB, numG)), denA)
[R,P,K] = sig.tf2zpk( numCL, denCL)
print('R_CLOSED: ', R, '\nP_CLOSED: ', P, '\nK_CLOSED: ', K)

tout , yout = sig.step(( numCL , denCL ) , T = t )

plt.subplot(1,2,2)
plt.plot(tout,yout)
plt.grid(True)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.title('Closed loop y(t)')
plt.show()



