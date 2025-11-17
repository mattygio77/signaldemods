import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

plt.style.use('dark_background')

sample_rate, audio_data = wavfile.read('./signals/QPSK/QPSK_IQ_Fs48KHz.wav')

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

plt.figure(figsize=(6, 4))
plt.plot(np.fft.fftshift(frequencies), np.fft.fftshift(np.abs(fft_result)))
plt.title('FFT Magnitude of IQ Data')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()