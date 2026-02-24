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

def costaLoop(signal, Kp=0.01, Ki=0.0001, numPasses=5):

    # normalize the signal to +-1
    signal /= np.sqrt(np.mean(np.abs(signal)**2)) + 1e-12

    N = len(signal)
    phase = 0.0
    integrator = 0.0
    sig_rot = np.zeros_like(signal, dtype=np.complex64)
    phase_hist = np.zeros(N)
    for pass_idx in range(numPasses):
        for n in range(N):
            v = signal[n] * np.exp(-1j * phase)
            I = v.real
            Q = v.imag

            s = 1.0 if I >= 0 else -1.0
            error = s * Q

            integrator += Ki * error
            phase += Kp * error + integrator

            if phase > np.pi or phase < -np.pi:
                phase = (phase + np.pi) % (2*np.pi) - np.pi
            sig_rot[n] = v
            phase_hist[n] = phase
    return sig_rot, phase_hist

sample_rate, audio_data = wavfile.read('./signaldemods/BPSK/BPSK-Decode/BPSK_IQ_Fs48KHz.wav')

print(f"Sample Rate: {sample_rate}")

i_samples = audio_data[:, 0]
q_samples = audio_data[:, 1]

# Compute FFT
complex_signal = i_samples + 1j * q_samples
fft_result = np.fft.fft(complex_signal)
frequencies = np.fft.fftfreq(len(complex_signal), d=1/sample_rate)

# Square law calc for coarse freq correction
square = complex_signal*complex_signal
squareFFT = np.fft.fft(square)
squareFreq = np.fft.fftfreq(len(square), d=1/sample_rate)
squareAbs = np.abs(squareFFT)
peakIdx = np.argmax(squareAbs)
print(f"Coarse Freq Correction: {squareFreq[peakIdx]/2}")

# Freq Adjust
freqOffset = squareFreq[peakIdx]/2
duration = len(i_samples)/sample_rate
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
complexConjugate = np.exp(-1j * (2 * np.pi * freqOffset * t))
freqAdjSig = complexConjugate * complex_signal

numPass = 1
phaseAdjSig, phaseHist = costaLoop(freqAdjSig, numPasses=numPass)
print(f"Settled Phase Correction: {np.mean(phaseHist[len(phaseHist)-5:])}")

# Phase Adjust Manual
# phaseOffset = np.pi * -.3
# complexConjugatePhase = np.exp(-1j * (2 * np.pi * freqOffset * t + phaseOffset))
# phaseAdjSig = complexConjugatePhase * complex_signal

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
plottingLib.plotFreq(complex_signal, sample_rate)
plottingLib.plotFreq(phaseAdjSig, sample_rate)
plottingLib.plotConstallation(freqAdjSig, phaseAdjSig)

plt.show()
