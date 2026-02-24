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

def qpskCostaLoop(sig, loopBw= 0.01, damping = 1/np.sqrt(2)):

    K = 1.0 # Loop Gain
    theta = loopBw / (damping + 1/(4* damping))
    d = (1 + 2 * damping * theta + theta**2)
    K1 = (4 * damping * theta)/d 
    K2 = (4 * theta**2)/d
    
    sig /= np.sqrt(np.mean(np.abs(sig)**2)) + 1e-12

    N = len(sig)
    phase = 0.0
    freq = 0.0

    sigRot = np.zeros_like(sig, dtype=np.complex64)
    phaseHist = np.zeros(N)
    errorHist = np.zeros(N)
    numPasses = 1

    for passIdx in range(numPasses):
        for n in range(N):
            v = sig[n] * np.exp(-1j * phase)
            I = v.real
            Q = v.imag

            iHat = np.sign(I) if I != 0 else 1
            qHat = np.sign(Q) if Q != 0 else 1

            error = I * qHat - Q * iHat

            freq = freq + K2 * error
            phase = phase + freq + K1 * error

            if phase > np.pi or phase < -np.pi:
                phase = (phase + np.pi) % (2*np.pi) - np.pi
            
            sigRot[n] = v
            phaseHist[n] = phase
            errorHist[n] = error
    
    return sigRot, phaseHist, errorHist


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
comConj = np.exp(-1j* (2 * np.pi* freqFix * t))
freqAdust = complex_signal * comConj

# Using Costa Loop for Fine Phase Correction
phaseAdj, phaseHist, errorHist = qpskCostaLoop(freqAdust)
print(f"Settled Phase Correction: {np.mean(phaseHist[len(phaseHist)-5:])}")

# Rotate back to quadrant add pi/2 to align the symbols
phaseAdj *= np.exp(1j * (np.pi/4+np.pi/2))

# Bit Decisions
sps = 40
bits = []
for n in range(int(phaseAdj.size/sps)):
    iDec = np.sum(np.real(phaseAdj)[n*sps:(n+1)*sps])
    qDec = np.sum(np.imag(phaseAdj)[n*sps:(n+1)*sps])
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

ans = []
for n in range(int(len(bits)/8)):
    byteList = bits[n*8:(n+1)*8]
    byteValue = int("".join(map(str, byteList)), 2)
    ans.append(chr(byteValue))
print(f"ans: {ans}")

#Plotting
plottingLib.plotTime(complex_signal, phaseAdj)
plottingLib.plotFreq(complex_signal, sample_rate, phaseAdj)
plottingLib.plotConstallation(freqAdust, phaseAdj)
plottingLib.plotConstallation(phaseAdj)

plt.show()
