#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

#Local imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import plottingLib

plt.style.use('dark_background')

sample_rate, audio_data = wavfile.read('./signaldemods/QPSK/QPSK-Decode/QPSK_IQ_Fs48KHz.wav')

print(f"Sample Rate: {sample_rate}")

i_samples = audio_data[:, 0]
q_samples = audio_data[:, 1]

complex_signal = i_samples + 1j * q_samples

sigFft = np.fft.fft(complex_signal)
sigFftAbs = np.abs(sigFft)
peakIdx = np.argmax(sigFftAbs)

freqs = np.fft.fftfreq(len(complex_signal), d=1/sample_rate)

print(f"freq with peak: {freqs[peakIdx]}")

freqOffset = freqs[peakIdx]
phaseOffset = 0
duration = len(i_samples)/sample_rate

t =np.linspace(0, duration, int(sample_rate*duration), endpoint=False)
complexConjugate = np.exp(-1j* (2 * np.pi* freqOffset * t + phaseOffset))

sigAdj = complex_signal * complexConjugate

#Plotting
plottingLib.plotTime(complex_signal, sigAdj)
plottingLib.plotFreq(complex_signal, sample_rate, sigAdj)
plottingLib.plotConstallation(sigAdj)

plt.show()
