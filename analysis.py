import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.filter_design import butter
from scipy.signal.signaltools import sosfilt
import soundfile as sf
import json

DEBUG = True
SAVE = True

f_name = str(input())
file_list = json.load(open(f_name + '.json', 'r'))
f_center = file_list['f_center']
bandwidth = file_list['bandwidth']
N = 2**12


for file in file_list['files'] :
    path = file['path']
    print("#---------------------------------------------#")
    print(path)
    full_audio, Fs = sf.read(path)
    filter = butter(4, [f_center-bandwidth, f_center+bandwidth],btype = 'bandpass', output = 'sos', fs = Fs)

    for t_start, t_stop in file['timestamps'] :

        print(f"{t_start = }")
        print(f"{t_stop = }")
        print(f"{Fs = }")

        if file['is_mono'] :
            chunk = full_audio[t_start*Fs : t_stop*Fs]
        else :
            chunk = full_audio[t_start*Fs : t_stop*Fs, 0]
            chunk += full_audio[t_start*Fs : t_stop*Fs, 1]

        filtered_chunk = sosfilt(filter, chunk)
        filtered_chunk /= filtered_chunk.std()

        f, t, Sxx = sig.spectrogram(filtered_chunk, Fs, nperseg = N, noverlap = 7*N // 8,scaling = 'spectrum')#, window = 'nuttall')
    

        # Variables for later use
        L = t.size
        dt = t[1] - t[0]

        if SAVE :
            plt.figure(figsize=(20,5))
            plt.pcolormesh(t,f,Sxx, shading = 'auto')
            plt.ylim(f_center-1.25*bandwidth, f_center+1.25*bandwidth)
            plt.xlabel("Time (in secs)")
            plt.ylabel("Frequency (in Hz)")
            plt.savefig(f"{path}_spectrogram.png")
        if DEBUG :
            plt.show()

        # Get the index corresonding to the orignial frequency
        center_idx = int(f_center*N/Fs)
        # Calculate the variation in the amplitude of all the frequecies over time
        sigma_f = np.std(Sxx, axis = 1)*np.abs(f-f_center)
        # sigma_f = np.max(Sxx, axis = 1)/np.mean(Sxx, axis = 1)

        #Get the index and frequencies that have the maximum deviation.
        freq_min_idx = np.argmax(sigma_f[:center_idx])
        freq_max_idx = center_idx + np.argmax(sigma_f[center_idx:])
        freq_min = f[freq_min_idx]
        freq_max = f[freq_max_idx]

        # Print said frequencies to use in further calculations
        print(f"{freq_min = }")
        print(f"{freq_max = }")

        if DEBUG :
            plt.plot(f, sigma_f)
            plt.scatter([freq_min, freq_max], [sigma_f[freq_min_idx], sigma_f[freq_max_idx]], color = 'red', marker = 'x')
            plt.show()

        # Get the time variation fo the amplitude of the frequencies of interest
        freq_min_slice = Sxx[freq_min_idx, :]
        freq_max_slice = Sxx[freq_max_idx, :]
        freq_min_slice /= freq_min_slice.std()
        freq_max_slice /= freq_max_slice.std()

        freq_freqs, freq_min_psd = sig.welch(freq_min_slice, fs = 1/dt, nperseg = 512, scaling='density')
        freq_freqs, freq_max_psd = sig.welch(freq_max_slice, fs = 1/dt,  nperseg = 512, scaling='density')

        freq_min_freq_idx = np.argmax(freq_min_psd)
        freq_max_freq_idx = np.argmax(freq_max_psd)

        if abs(freq_min_freq_idx - freq_max_freq_idx) > 1 :
            print("SIGNIFICANT ERROR IN TIME PERIOD")
            print("Time Period MISMATCH, please check time period manually")

        if DEBUG :
            plt.plot(t, freq_min_slice)
            plt.plot(t, freq_max_slice)
            plt.scatter(freq_freqs[freq_min_freq_idx], freq_min_psd[freq_min_freq_idx], color = 'red', marker = 'x')
            plt.scatter(freq_freqs[freq_max_freq_idx], freq_max_psd[freq_max_freq_idx], color = 'red', marker = 'x')
            plt.show()

            plt.plot(freq_freqs, freq_min_psd)
            plt.plot(freq_freqs, freq_max_psd)
            plt.scatter(freq_freqs[freq_min_freq_idx], freq_min_psd[freq_min_freq_idx], color = 'red', marker = 'x')
            plt.scatter(freq_freqs[freq_max_freq_idx], freq_max_psd[freq_max_freq_idx], color = 'red', marker = 'x')
            plt.show()

        f_fan = 0.5*(freq_freqs[freq_min_freq_idx] + freq_freqs[freq_max_freq_idx])
        w_fan = 2*np.pi*f_fan
        print(f'{f_fan = }')
        print(f'{w_fan = }')
