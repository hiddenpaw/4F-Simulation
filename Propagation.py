#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image


# In[3]:


def prop(image,distance):
    pitch=7.56e-6
    wavelength=633e-9



    k=2*np.pi/wavelength

    #image=imageio.imread('44.bmp')+0j
    size=image.shape
    length=pitch*size[0]
    width=pitch*size[1]

    prop_mask=np.zeros((size[0],size[1]))
    kx_max=2*np.pi/length*size[0]
    ky_max=2*np.pi/width*size[1]
    kx_array=np.linspace(0,kx_max,size[0]).reshape(size[0],1)
    ky_array=np.linspace(0,ky_max,size[1]).reshape(1,size[1])
    prop_mask=np.exp(-1j*(kx_array**2+ky_array**2)*distance/2/k)
    fft=np.fft.fft2(image)
    leak_mask=1
    #leak_mask=(1-2*kx_array*distance/k/width)*(1-2*ky_array*distance/k/length)
    output=np.fft.ifft2(prop_mask*fft*leak_mask)
    
    return output

def prop_angle(image,a):
    #angle in degree
    
    pitch=7.56e-6
    wavelength=633e-9

    sin=np.sin(a/180*np.pi)
    cos=np.cos(a/180*np.pi)

    k=2*np.pi/wavelength

    #image=imageio.imread('44.bmp')+0j
    size=image.shape
    length=pitch*size[0]
    width=pitch*size[1]
    
    fft=np.fft.fft2(image)
    fft_new=np.zeros((size[0],size[1]))*1j
    
    for i in range(size[0]):
        for j in range(size[1]):
            if i==0 and j==0:
                fft_new[i,j]=fft[i,j]
            else:
                fx=fft[i,j]*i/np.sqrt(i**2+j**2)
                fy=fft[i,j]*j/np.sqrt(i**2+j**2)
                #fft_new[i,j]=np.sqrt(fx**2+fy**2)
                fft_new[i,j]=np.sqrt((cos*fx-np.pi*sin/k*(fx**2/length))**2+fy**2)
                if fft[i,j].real>0:
                    fft_new[i,j]=complex(np.abs(fft_new[i,j].real), fft[i,j].imag) 
                else:
                    fft_new[i,j]=complex(-np.abs(fft_new[i,j].real), fft[i,j].imag)
                if fft[i,j].imag>0:
                    fft_new[i,j]=complex(fft_new[i,j].real, np.abs(fft[i,j].imag))
                else:
                    fft_new[i,j]=complex(fft_new[i,j].real, -np.abs(fft[i,j].imag))
                    
                


    output=np.fft.ifft2(fft_new)
    
    return output
    


# In[ ]:




