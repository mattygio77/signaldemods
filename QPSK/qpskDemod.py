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
duration = len(i_samples)/sample_rate

# Course Freq Correction
squLaw = complex_signal * complex_signal
fourLaw = squLaw * squLaw

fourLawFft = np.abs(np.fft.fft(fourLaw))
peakIdx = np.argmax(fourLawFft)
freqs = np.fft.fftfreq(len(fourLaw), d=1/sample_rate)
freqFix = freqs[peakIdx]/4
print(f"Freq Offset is: {round(freqFix, 2)} Hz")
t = np.linspace(0, duration, int(sample_rate*duration), endpoint=False)

# QPSK is rotated. Try phase correction for fix
phase = np.pi/2
# phase = 0
comConj = np.exp(-1j* (2 * np.pi* freqFix * t + phase))

freqAdust = complex_signal * comConj

sps = 40

bits = []
for n in range(int(freqAdust.size/sps)):
    iDec = np.sum(np.real(freqAdust)[n*sps:(n+1)*sps])
    qDec = np.sum(np.imag(freqAdust)[n*sps:(n+1)*sps])
    if iDec > 0 and qDec > 0: # 45 deg
        bits.extend([0, 0])
    elif iDec < 0 and qDec > 0: # 135 Deg
        bits.extend([0, 1])
    elif iDec > 0 and qDec < 0: # -45 Deg
        bits.extend([1, 0])
    elif iDec < 0 and qDec < 0: # -135 Deg
        bits.extend([1, 1])
    else: 
        print("Error with Bit Decisions. You should never see")
print(f"Bits: {bits}")
print(f"len of bits: {len(bits)}")

ans = []
for n in range(int(len(bits)/8)):
    byteList = bits[n*8:(n+1)*8]
    byteValue = int("".join(map(str, byteList)), 2)
    ans.append(chr(byteValue))
print(f"ans: {ans}")

#Plotting
plottingLib.plotTime(complex_signal, freqAdust)
plottingLib.plotFreq(complex_signal, sample_rate, freqAdust)
# plottingLib.plotConstallation(complex_signal, freqAdust)
plottingLib.plotConstallation(freqAdust)

plt.show()
