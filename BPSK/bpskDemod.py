#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

plt.style.use('dark_background')

sample_rate, audio_data = wavfile.read('./BPSK_IQ_Fs48KHz.wav')

print(f"Sample Rate: {sample_rate}")

i_samples = audio_data[:, 0]
q_samples = audio_data[:, 1]

fig1, axs1 = plt.subplots(2, 1)
axs1[0].plot(i_samples, label="In-phase (I)")
axs1[0].plot(q_samples, label="Quadrature (Q)")
axs1[0].set_xlabel("Sample Index")
axs1[0].set_ylabel("Amplitude")
axs1[0].set_title("I/Q Time-Domain Plot")
axs1[0].legend()
axs1[0].grid(True)

complex_signal = i_samples + 1j * q_samples
fft_result = np.fft.fft(complex_signal)
frequencies = np.fft.fftfreq(len(complex_signal), d=1/sample_rate)

fig, axs = plt.subplots(2, 1)
axs[0].plot(np.fft.fftshift(frequencies), np.fft.fftshift(np.abs(fft_result)))
axs[0].set_title('FFT Magnitude of IQ Data')
axs[0].set_xlabel('Frequency (Hz)')
axs[0].set_ylabel('Magnitude')

freqOffset = 4320
phaseOffset = np.pi * -.2
duration = len(i_samples)/sample_rate
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
complexConjugate = np.exp(-1j * (2 * np.pi * freqOffset * t))
freqAdjSig = complexConjugate * complex_signal

complexConjugatePhase = np.exp(-1j * (2 * np.pi * freqOffset * t + phaseOffset))
phaseAdjSig = complexConjugatePhase * complex_signal

fft_result = np.fft.fft(freqAdjSig)
frequencies = np.fft.fftfreq(len(freqAdjSig), d=1/sample_rate)

axs[1].plot(np.fft.fftshift(frequencies), np.fft.fftshift(np.abs(fft_result)))
axs[1].set_title('FFT Magnitude of IQ Data')
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Magnitude')
plt.tight_layout()

freqAdjSigI = np.real(freqAdjSig)
freqAdjSigQ = np.imag(freqAdjSig)

axs1[1].plot(np.real(phaseAdjSig), label="In-phase (I)")
# axs1[1].plot(np.imag(phaseAdjSig), label="Quadrature (Q)")
axs1[1].set_xlabel("Sample Index")
axs1[1].set_ylabel("Amplitude")
axs1[1].set_title("I/Q Time-Domain Plot")
axs1[1].legend()
axs1[1].grid(True)
# plt.show() 

limits = 30e3
plt.figure(figsize=(5, 5))
plt.scatter(freqAdjSigI, freqAdjSigQ, s=10, alpha=0.7, label="Phase Offset")
plt.title('IQ Constellation Diagram')
plt.xlabel('In-phase (I)')
plt.ylabel('Quadrature (Q)')
plt.grid(True)
plt.axhline(0, color='gray', linewidth=4)
plt.axvline(0, color='gray', linewidth=4)
plt.xlim(-limits, limits)
plt.ylim(-limits, limits)

plt.scatter(np.real(phaseAdjSig), np.imag(phaseAdjSig), s = 10, alpha=0.7, label="Phase Adjusted")
plt.legend()

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
plt.show()
