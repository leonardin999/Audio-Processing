# -*- coding: utf-8 -*-
"""
Created on Fri May 28 09:06:44 2021

@author: Leonard
"""

from module_function import *
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import glob
import pathlib
from IPython import display
from scipy.io import loadmat,savemat
import pandas as pd
from matplotlib import cm # colour map
import plotly.graph_objects as go
# (rate,sig) = wav.read("0ea0e2f4_nohash_1.wav")
# [mfcc_feat,feat] = mfcc(sig, rate)

# plt.plot(np.linspace(0, len(mfcc_feat), num=len(mfcc_feat)), mfcc_feat)    
data_dir = pathlib.Path('mini_speech_commands_Auto')
if not data_dir.exists():
    print('direction not found!')
commands = np.array(os.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
filenames = glob.glob(str(data_dir)+'/*/*')
pathfiles = glob.glob(str(data_dir)+'/*')
num_samples = len(filenames)
print('Number of total examples:', num_samples)
print('Number of examples per label:',
      len(os.listdir(str(data_dir/commands[0]))))
print('Example file:', filenames[0])
def get_label(file_path):
    parts=[]
    for i in range(np.size(file_path)):
        parts_take = np.array([file_path[i].split('\\')])
        parts =np.append(parts,parts_take[0,1])                                
    return parts
## create folder to save file
dirName = 'input_data'
if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("Directory " , dirName ,  " Created ")
else:    
    print("Directory " , dirName ,  " already exists")  
        
# data__input_dir = pathlib.Path('input_data')
# mfcc_1=[]
# for i in range(len(pathfiles)):
#     mfcc_1[i] = mutiple_file(pathfiles[i])
# #savemat(str(data__input_dir)+"\\"+commands[0]+'.mat',{str(commands[0]):mfcc_down})
mfcc_result=[]
#folder = str(folder).strip('\u202a')
listname = os.listdir(pathfiles[0]) # dir is your directory path
number_files = len(listname)

data_dir = pathlib.Path('mini_speech_commands_Auto')
if not data_dir.exists():
    print('direction not found!')
commands = np.array(os.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
filenames = glob.glob(str(data_dir)+'/*/*')
pathfiles = glob.glob(str(data_dir)+'/*')
#filenames = random.shuffle(filenames)
num_samples = len(filenames)
print('Number of total examples:', num_samples)
print('Number of examples per label:',
      len(os.listdir(str(data_dir/commands[0]))))
print('Example file:', filenames[0])
def get_label(file_path):
    parts=[]
    for i in range(np.size(file_path)):
        parts_take = np.array([file_path[i].split('\\')])
        parts =np.append(parts,parts_take[0,1])                                
    return parts
label = get_label(filenames)
#(sr1,signal1) = read_wavfile('D:\ONLINE_10_CLASS_LEARNING\INPUT_SOUND_WAV_4\\01_NO\\30_no.wav')
(sr,signal) = read_wavfile(filenames[90])
mfcc_feat = feature_extract(signal, sr)
winlen=0.025
winstep=0.01
Frame_len = winlen*sr
FrameStep = winstep*sr
num_frames = int(np.ceil(float(np.abs(len(signal) - Frame_len)) / FrameStep))
cepstral_coefficents,pspec,feat,log_bank,frames = mfcc(signal,sr,winlen,winstep,numcep=20,
         nfilt=28,nfft=400,lowfreq=0,highfreq=None,fcut = 3000,ceplifter=22,open_lift = True,
         winfunc="none",sum_up = True,appendEnergy=True)

tfrm = np.arange(Frame_len/2,Frame_len/2+(num_frames)*FrameStep,FrameStep,dtype=float)/sr
rows,cols= np.shape(pspec)
F = np.linspace(0,(sr)*(1-1/cols)/1000,cols)
tt = tfrm[:,None]
plt.ion()
plt.style.use("seaborn-notebook")
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')  # set the axes for 3D plot
surf = ax.plot_surface( tfrm[:, None],F[None,:],10.0*np.log10(pspec), cmap=cm.jet,linewidth=5)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (Herzt)')
ax.set_zlabel('Amplitude (Db)')
plt.draw()
plt.show()
fig2 = plt.figure()
mesh = plt.pcolormesh(tfrm,F,(10.0*np.log10(pspec)).T, cmap=cm.jet,linewidth=5)
plt.pcolormesh(tfrm,F,(10.0*np.log10(pspec)).T, cmap=cm.jet,linewidth=5)
fig2.colorbar(mesh).set_label('Intensity [dB]',fontsize=20)
plt.xlabel('time (in seconds)',fontsize=20)
plt.ylabel('frequency',fontsize=20)
plt.show()

fig3 = plt.figure()
time = np.linspace(0,2,len(signal))
plt.plot(time,signal,linewidth=1)
plt.legend(['Hình vuông'],fontsize=20)
plt.xlabel('time (in seconds)',fontsize=20)
plt.ylabel('amplitude',fontsize=20)
plt.grid(True)
plt.show()

fig4 = plt.figure()
x_ax= np.linspace(1,20,len(mfcc_feat))
plt.plot(x_ax,mfcc_feat,linewidth=1,color='red')
plt.legend(['Hình vuông'],fontsize=20)
plt.xlabel('samples',fontsize=20)
plt.ylabel('amplitude',fontsize=20)
plt.grid(True)
plt.show()

fig5 = plt.figure()
fig5.suptitle('Signals at Frames No.50th')
plt.plot(frames[50,:],linewidth=1,color='k')
plt.legend(['Hình vuông'],fontsize=20)
plt.xlabel('samples',fontsize=20)
plt.ylabel('amplitude',fontsize=20)
plt.grid(True)
plt.show()


