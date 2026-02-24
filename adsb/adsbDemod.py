#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

#Local imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import plottingLib

plt.style.use('dark_background')

sample_rate, audio_data = wavfile.read('./signaldemods/adsb/adsb.2021-11-26T15_03_30_573.wav')

print(f"Sample Rate: {sample_rate}")
print(f"Length of data: {len(audio_data)}")

i_samples = audio_data[:1024*100, 0]
q_samples = audio_data[:1024*100, 1]

# Compute FFT
complex_signal = i_samples + 1j * q_samples
fft_result = np.fft.fft(complex_signal)
frequencies = np.fft.fftfreq(len(complex_signal), d=1/sample_rate)

# plottingLib.plotTime(complex_signal)
# plottingLib.plotFreq(complex_signal, sample_rate)

# Batch Processing since its a big file
batchSize = 1024 * 100
avgSize = 1000
multiplier = 5.0
for idx in range(0, len(audio_data), batchSize):
    i_samples = audio_data[idx:idx+batchSize, 0]
    q_samples = audio_data[idx:idx+batchSize, 1]
    samps = i_samples + 1j * q_samples

    magnitudes = np.sqrt(samps[:avgSize].real**2+samps[:avgSize].imag**2)

    # Not really that usuefully unless Mags is longer
    # movingAvg = np.convolve(magnitudes, np.ones(avgSize)/avgSize, mode="same")
    # print(f"Length of moving Avg: {len(movingAvg)}")
    # threshold = np.mean(movingAvg) * multiplier

    # Going to try random sample indexing
    sample_indices = np.random.choice(len(magnitudes), size=1000, replace=False)
    threshold = np.mean(magnitudes[sample_indices]) * multiplier
    print(f"Threshold: {threshold}")

    detections = magnitudes > threshold
    print(f"Detections: {detections}")


plottingLib.plotTime(samps)
plottingLib.plotFreq(samps, sample_rate)

plt.show()