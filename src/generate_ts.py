import numpy as np
import matplotlib.pyplot as plt

def get_random_ts(filename = None):

    if filename is not None:
        TS = np.loadtxt(filename)
        return TS
    """
    " All constants declaration
    """
    freq_ranges = [[1.5, 3.5],  # delta Unconscious, deep sleep
                   [3.5, 7.5],  # theta Reduced consciousnenss
                   [7.5, 12.5],  # alpha Physical and mental relaxation
                   [12.5, 19.5]]  # beta Engaged mind
                
    #freq_ranges = [[1.5, 3.5]]

    sample_rate = 50              
    duration_of_segment = 10 #(in seconds)
    num_segments = len(freq_ranges)

    Time = np.linspace(0,  duration_of_segment * num_segments, num_segments * duration_of_segment * sample_rate)

    sliding_window_size = 50

    frequencies = []
    for freq in freq_ranges:
        frequencies.append(np.random.uniform(freq[0], freq[1]))
        
    frequencies = np.array(frequencies)
    rad_frequencies = 2*np.pi*frequencies
        
    stds = [1, 2, 5, 7]

    """
    " END constants declaration
    """

    TS = []
    for i, freq in enumerate(frequencies):
        std = stds[i:]+stds[:i]
        rvals = np.random.normal(0, 0.5, num_segments)
        
        #rvals = np.ones(4)
        
        start = i*duration_of_segment * sample_rate 
        
        time = Time[start : start + duration_of_segment * sample_rate]
        #print(time.shape)
        
        delta = np.random.normal(0, std[0], time.shape[0]) * rvals[0]
        theta = np.random.normal(0, std[1], time.shape[0]) * rvals[1]
        alpha = np.random.normal(0, std[3], time.shape[0]) * rvals[2]
        beta = np.random.normal(0, std[2], time.shape[0]) * rvals[3]
        
        
        #delta = theta = alpha = beta = np.ones((time.shape[0]))
        
        delta_signal = delta * np.cos(rad_frequencies[0] * time) + delta * np.sin(rad_frequencies[0] * time)
        theta_signal = theta * np.cos(rad_frequencies[1] * time) + theta * np.sin(rad_frequencies[1] * time)
        alpha_signal = alpha * np.cos(rad_frequencies[2] * time) + alpha * np.sin(rad_frequencies[2] * time)
        beta_signal = beta * np.cos(rad_frequencies[3] * time) + beta * np.sin(rad_frequencies[3] * time)
        
        delta_signal = alpha_signal = beta_signal = np.zeros(delta_signal.shape[0])
        
        TS = np.hstack((TS, delta_signal + theta_signal + alpha_signal + beta_signal))

    return TS
