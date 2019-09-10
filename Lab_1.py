#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Example 1
import math as m
import numpy as np
print(np.pi)


# In[2]:


#Example 2
t=2**7
print(t)
print("t=", t, "second")


# In[3]:


#Example 3
array1=np.array([0,1,2,3])
print('array1:',array1)
array2=np.array([[0],[1],[2]])
print('array2:')
print(array2)
array3=np.array([[0,1],[1,2]])
print('array3:')
print(array3)


# In[4]:


#Example 4
print('1x3',np.zeros(3))
print('1x3',np.ones(3))


# In[6]:


#Example 5
import numpy as np
import matplotlib.pyplot as plt

#Variables
steps=0.1
x=np.arange(-2,2+steps,steps)

y1=x+2
y2=x**2

#Code for plots
plt.figure(figsize=(12,8))           #custom figure size
plt.title('Sample Plots for Lab 1')  #Figure title

plt.subplot(3,1,1)                   #subplot 1
plt.plot(x,y1)
plt.ylabel('Subplot 1')
plt.grid(True)

plt.subplot(3,1,2)                   #subplot 2
plt.plot(x,y2)
plt.ylabel('Subplot 2')
plt.grid(which='both')

plt.subplot(3,1,3)                   #subplot 3
plt.plot(x,y1,'--r', label='y1')
plt.plot(x,y2,'o', label='y2')
plt.axis([-2.5,2.5,-0.5,4.5])
plt.grid(True)
plt.legend(loc='lower right')
plt.xlabel('x')
plt.ylabel('Subplot 3')

plt.show


# In[7]:


#Example 6
import numpy as np

cRect = 2 + 3j
print(cRect)

cPol = abs(cRect) * np.exp(1j*np.angle(cRect))
print(cPol)

cRect2 = np.real(cPol) + 1j*np.imag(cPol)
print(cRect2)


# In[ ]:




