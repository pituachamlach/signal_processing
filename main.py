import numpy as np
from scipy.signal import find_peaks ,resample_poly,spectrogram
import matplotlib.pyplot as plt
import math
import h5py
import torch
SDR_FS=15.36e6


def calculate_threshold(Energy_signal):
    hist_ = np.histogram(Energy_signal)  
    hist_0 = hist_[0]
    hist_1 = hist_[1]
    sum_ = 0
    mean = 0
    m2 = 0
    
    for i in range(len(hist_0)):
        sum_ += hist_0[i]
    
    hist_0 = hist_0/sum_
    for i in range(len(hist_0)):
        mean += hist_0[i]*hist_1[i]
    
    for i in range(len(hist_0)):
        m2 += hist_0[i]*(hist_1[i]**2)
    std = math.sqrt(m2 - mean**2)
    
    return mean, std
    

def signal_separation(raw_data):
    # convert raw_data to binary signal and after that find peaks
    # return peaks and their properties
    
    raw_data = raw_data[round(200e-3*15.36e6):round(225e-3*15.36e6)]
    energy_signal = np.abs(np.array(raw_data))**2
    threshold, std=calculate_threshold(energy_signal)
    above_level_noise = energy_signal >= threshold
    kernel = np.ones(1500) / 1500  # Averaging over a window of size 5

# Apply convolution to filter the array
    filtered_array = np.convolve(above_level_noise, kernel, mode='valid') # plt.plot(above_level_noise)
    filtered_array= filtered_array >= threshold
    T_max_time_domain = 3.4e-3
    T_min_time_domain = 0.18e-3
    T_max_samples = round(T_max_time_domain*SDR_FS)
    T_min_samples = round(T_min_time_domain*SDR_FS)
    peaks, properties = find_peaks(filtered_array, width=[T_min_samples, T_max_samples])
    plt.plot(filtered_array)
    th=np.full(fill_value=threshold,shape=above_level_noise.shape)
    plt.plot(th,color='black')
    plt.plot(energy_signal)
    plt.show()
    plt.plot(abs(raw_data))
    window_signal = np.zeros((1, len(raw_data)))
    for i in range(len(peaks)):
        window_signal[0][properties['left_bases'][i]:properties['right_bases'][i]] = np.max(abs(raw_data[properties['left_bases'][i]:properties['right_bases'][i]]))
    plt.plot(window_signal[0][:], 'r')
    plt.show()
    
    signals=[]
    for i in range(len(peaks)):
        signals.append(raw_data[properties['left_bases'][i]:properties['right_bases'][i]])
    
    return signals


def resample(raw_signal, fs_prev):
    # resample signal to 15.36Mhz zero-phase low-pass FIR filter is applied"
    resample_ratio = SDR_FS / fs_prev
    resampled_signal = resample_poly(raw_signal, up=1, down=resample_ratio)
    return resampled_signal

def zero_padding(signal):
    
    # add zero to end of signal
    num_zeros=round(15.36e6*8e-3)-signal.shape[0]
    signal_padded = np.pad(signal, (0, num_zeros), mode='constant')
    # S = signal.shape[0]
    # signal_after_padding[0][0:S] = signal[0:S]
    return signal_padded

def signal_normalization(signal):
    # convert the signal to with mean=0 and std=1
    mean_signal=np.mean(signal)
    std_signal=np.std(signal)
    normalized_signal=(signal-mean_signal)/std_signal
    return normalized_signal
    

def spectrogram_signal(signal, window_size_T=0.01e-3):
    nfft_ =  2**(math.ceil(math.log2(window_size_T*SDR_FS)))
    f, t, Sxx = spectrogram(signal.flatten(), SDR_FS, nfft=nfft_)
    W = np.shape(Sxx)[0]
    H = np.shape(Sxx)[1]
    print(W,H)
    spec_re_im = np.zeros((W, H, 2))
    spec_re_im[:, :, 0] = np.real(Sxx)
    spec_re_im[:, :, 1] = np.imag(Sxx)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    return spec_re_im

def real_time_signal_separation(raw_data):
    # raw_data:signal (0.7ms,10e6 samples) return spectrograms signals
    # signal separation
    signals=signal_separation(raw_data)
    #should be multi threaded.
    for signal in signals:
        # zero_padding
        signal_after_padding=zero_padding(signal)
        # norm
        norm_signal=signal_normalization(signal_after_padding)
        # spectrogram
        spectrogram=spectrogram_signal(norm_signal)
        # send spectrogram to model for prediction 
        
    
def dataset_processing_noisy(dataset_path):
    """ resample,padding,normalization,spectrogram"""
    #load dataset
    dataset_dict = torch.load(dataset_path + 'dataset.pt')
    x_iq = dataset_dict['x_iq']
    y = dataset_dict['y']
    spec_dataset_list=[]
    fs_prev = 14e6
    W,H = 256 ,548
    data_np = np.empty((len(x_iq), H, W))  
    for idx in range(len(x_iq)):
        signal = x_iq[idx] 
        resampled_sample=resample(signal,fs_prev=fs_prev,fs_next=SDR_FS)
        padded_sample=zero_padding(resampled_sample)
        norm_sample=signal_normalization(padded_sample)
        spec_signal=spectrogram_signal(norm_sample, fs=SDR_FS)
        data_np[idx]=spec_signal
    
    dataset_np=np.array(spec_dataset_list)
    file_path = 'noisy_spec_dataset.h5'
    with h5py.File(file_path, 'w') as hdf_file:
        hdf_file.create_dataset('noisy_spec', data=dataset_np)
        hdf_file.create_dataset('noisy_label', data=y)

if __name__ == '__main__':
    
    raw_data_path = 'Mavic_pro_1'
    f= open(raw_data_path, 'rb')
    complex_raw=np.fromfile(f, dtype='complex64')
    sig = signal_separation(complex_raw)
    sig = sig[0]
    signal_after_padding=zero_padding(sig)
    norm_signal=signal_normalization(signal_after_padding)
    spec = spectrogram_signal(norm_signal)
    
    