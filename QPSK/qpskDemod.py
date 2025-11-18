#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

plt.style.use('dark_background')

sample_rate, audio_data = wavfile.read('./QPSK_IQ_Fs48KHz.wav')

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
frequencies = np.fft.fftfreq(len(complex_signal), d=1/sample_rate)/1e3

plt.figure(figsize=(6, 4))
plt.plot(np.fft.fftshift(frequencies), np.fft.fftshift(np.abs(fft_result)))
plt.title('FFT Magnitude of IQ Data')
plt.xlabel('Frequency (KHz)')
plt.ylabel('Magnitude')

freqOffset = -2500 +71
phaseOffset = 0
duration = len(i_samples)/sample_rate

t =np.linspace(0, duration, int(sample_rate*duration), endpoint=False)
complexConjugate = np.exp(-1j* (2 * np.pi* freqOffset * t + phaseOffset))

sigAdj = complex_signal * complexConjugate

axs1[1].plot(np.real(sigAdj), label="In-phase(I)")
axs1[1].plot(np.imag(sigAdj), label="Quadrature (Q)")
axs1[1].set_xlabel("Sample Index")
axs1[1].set_ylabel("Amplitude")
axs1[1].set_title("Adjusted I/Q Time-Domain Plot")
axs1[1].legend()
axs1[1].grid(True)
fig1.tight_layout()

fftAdj = np.fft.fft(sigAdj)
freqAdj = np.fft.fftfreq(len(sigAdj), d=1/sample_rate)/1e3
plt.plot(np.fft.fftshift(freqAdj), np.fft.fftshift(np.abs(fftAdj)))
plt.tight_layout()

limits = 30e3
plt.figure(figsize=(5,5))
plt.scatter(np.real(complex_signal), np.imag(complex_signal), s=10, alpha=0.7, label="Original Signal")
plt.title("IQ Constellation Diagram")
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.grid(True)
plt.axhline(0, color="gray", linewidth=4)
plt.axvline(0, color="gray", linewidth=4)
plt.xlim(-limits, limits)
plt.ylim(-limits, limits)

plt.scatter(np.real(sigAdj), np.imag(sigAdj), s=10, alpha=0.7, label="Adjusted Signal")
plt.legend()

plt.show()


