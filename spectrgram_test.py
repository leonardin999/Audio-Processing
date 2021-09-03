# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 22:20:21 2021

@author: Leonard
"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy import signal # spectrogram function
from matplotlib import cm # colour map

# basic config
sample_rate = 11240.  # 
sig_len_secs = 10
frequency = 2000.

# generate the signal
timestamps_secs = np.arange(sample_rate*sig_len_secs) / sample_rate
mysignal = np.sin(2.0 * np.pi * frequency * timestamps_secs) 

# extract the spectrum
freq_bins, timestamps, spec = signal.spectrogram(mysignal, sample_rate)

# 3d plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(freq_bins[:, None], timestamps[None, :], 10.0*np.log10(spec), cmap=cm.coolwarm)
plt.show()