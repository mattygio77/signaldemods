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

sample_rate, audio_data = wavfile.read('./BPSK-Decode/BPSK_IQ_Fs48KHz.wav')

print(f"Sample Rate: {sample_rate}")

i_samples = audio_data[:, 0]
q_samples = audio_data[:, 1]

# Compute FFT
complex_signal = i_samples + 1j * q_samples
fft_result = np.fft.fft(complex_signal)
frequencies = np.fft.fftfreq(len(complex_signal), d=1/sample_rate)

# Freq Adjust
freqOffset = 4320
duration = len(i_samples)/sample_rate
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
complexConjugate = np.exp(-1j * (2 * np.pi * freqOffset * t))
freqAdjSig = complexConjugate * complex_signal

# Phase Adjust
phaseOffset = np.pi * -.2
complexConjugatePhase = np.exp(-1j * (2 * np.pi * freqOffset * t + phaseOffset))
phaseAdjSig = complexConjugatePhase * complex_signal

fft_result = np.fft.fft(freqAdjSig)
frequencies = np.fft.fftfreq(len(freqAdjSig), d=1/sample_rate)

freqAdjSigI = np.real(freqAdjSig)
freqAdjSigQ = np.imag(freqAdjSig)

# Decoding Frame
spb = 40
intDecision = []
for n in range(int(phaseAdjSig.size/spb)):
    intDecision.append(np.sum(np.real(phaseAdjSig)[n*spb:(n+1)*spb]))
bits = (np.real(intDecision) > 0).astype(int)

ans = []

for n in range(int(len(bits)/8)):
    byteList = bits[n*8:(n+1)*8]
    byte_value = int("".join(map(str, byteList)), 2)
    ans.append(chr(byte_value))

print(f"ans: {ans}")

#Plotting
plottingLib.plotTime(complex_signal, phaseAdjSig)
plottingLib.plotFreq(complex_signal, sample_rate, phaseAdjSig)
plottingLib.plotConstallation(freqAdjSig, phaseAdjSig)

plt.show()
