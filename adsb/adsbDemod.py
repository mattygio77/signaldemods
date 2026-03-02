#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import time

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

preamble_template = np.array([
    1, 1,        # pulse at 0 μs (samples 0-1)
    -1,          # gap
    1, 1,        # pulse at 1.0 μs (samples 3-4)
    -1, -1, -1,  # gap
    1, 1,        # pulse at 3.5 μs (samples 8-9)
    -1,          # gap
    1, 1,        # pulse at 4.5 μs (samples 11-12)
    -1, -1, -1, -1, -1, -1  # trailing gap
])

# Batch Processing since its a big file
batchSize = 1024 * 100
avgSize = 1000
multiplier = 5.0

# Checking Processing Time
start = time.time()
totSamps = int(len(audio_data)/20)

for idx in range(0, totSamps, batchSize):
    i_samples = audio_data[idx:idx+batchSize, 0]
    q_samples = audio_data[idx:idx+batchSize, 1]
    samps = i_samples + 1j * q_samples

    magnitudes = np.sqrt(samps.real**2+samps.imag**2)

    movingAvg = np.convolve(magnitudes, np.ones(avgSize)/avgSize, mode="same")
    threshold = movingAvg * multiplier

    # detections = magnitudes > threshold
    # print(f"Detections: {detections}")
    
    magsNorm = magnitudes / threshold
    correlation = np.correlate(magsNorm, preamble_template, mode="valid")

    correlationThreshold = np.percentile(correlation, 99)
    peaks = np.where(correlation > correlationThreshold)[0]

    min_separation = 134
    preambles = []
    for peak in peaks:
        if not preambles or (peak - preambles[-1]) > min_separation:
            preambles.append(peak)


    # Going to try random sample indexing. Might need if we slow down.
    # magnitudes = np.sqrt(samps[:avgSize].real**2+samps[:avgSize].imag**2)
    # sample_indices = np.random.choice(len(magnitudes), size=1000, replace=False)
    # threshold = np.mean(magnitudes[sample_indices]) * multiplier
    # print(f"Threshold: {threshold}")

# Real Time Check
elapsed = time.time() - start
print(f"Elapsed Time: {elapsed}")
print(f"Duration of Data: {totSamps / sample_rate}")
if totSamps / sample_rate > elapsed:
    print("[Pass] We are in real time")
else: 
    print("[Fail] We are slower than real time")

plottingLib.plotTime(samps)
plottingLib.plotFreq(samps, sample_rate)

plt.show()