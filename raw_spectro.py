import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import json

t_start = 60
t_stop = 70
N = 2**13

full_audio, Fs = sf.read('./datasets/RJ/RJ_55.wav')

chunk = full_audio[t_start*Fs : t_stop*Fs, 0]
chunk += full_audio[t_start*Fs : t_stop*Fs, 1]

f, t, Sxx = sig.spectrogram(chunk, Fs, nperseg = N, noverlap = 3*N // 4,window = 'nuttall')

plt.figure(figsize = (15,10))
plt.pcolormesh(t,f, Sxx, shading = 'auto')
#plt.ylim(10000, 12000)
plt.xlabel("Time (in secs)")
plt.ylabel("Frequency (in Hz)")
# plt.savefig(f'./datasets/{p_list[2]}/{p_list[3]}_spectrum.png')
plt.show()
