#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np


def plotConstallation(sig1, sig2=None):

    limits = 30e3
    plt.figure(figsize=(5,5))
    plt.scatter(np.real(sig1), np.imag(sig1), s=10, alpha=0.7, label="Original Signal")
    plt.title("IQ Constellation Diagram")
    plt.xlabel("In-Phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True)
    plt.axhline(0, color="gray", linewidth=4)
    plt.axvline(0, color="gray", linewidth=4)
    plt.xlim(-limits, limits)
    plt.ylim(-limits, limits)


    if sig2 is not None:
        plt.scatter(np.real(sig2), np.imag(sig2), s=10, alpha=0.7, label="Adjusted Signal")
        plt.legend()


# Expects I/Q for both signals
def plotFreq(sig1, sample_rate, sig2=None):

    fftResult1 = np.fft.fft(sig1)
    freqs1 = np.fft.fftfreq(len(sig1), d=1/sample_rate)/1e3

    plt.figure(figsize=(6, 4))
    plt.plot(np.fft.fftshift(freqs1), np.fft.fftshift(np.abs(fftResult1)))
    plt.title('FFT Magnitude of IQ Data')
    plt.xlabel('Frequency (KHz)')
    plt.ylabel('Magnitude')

    if sig2 is not None:
        fftResult2 = np.fft.fft(sig2)
        freqs2 = np.fft.fftfreq(len(sig2), d=1/sample_rate)/1e3
        plt.plot(np.fft.fftshift(freqs2), np.fft.fftshift(np.abs(fftResult2)))
        
    plt.tight_layout()


def plotTime(sig1, sig2=None):

    if sig2 is not None:
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(np.real(sig1), label="In-phase (I)")
        axs[0].plot(np.imag(sig1), label="Quadrature (Q)")
        axs[0].set_xlabel("Sample Index")
        axs[0].set_ylabel("Amplitude")
        axs[0].set_title("I/Q Time-Domain Plot")
        axs[0].legend()
        
        axs[0].grid(True)
        axs[1].plot(np.real(sig2), label="In-phase(I)")
        axs[1].plot(np.imag(sig2), label="Quadrature (Q)")
        axs[1].set_xlabel("Sample Index")
        axs[1].set_ylabel("Amplitude")
        axs[1].set_title("Adjusted I/Q Time-Domain Plot")
        axs[1].legend()
        axs[1].grid(True)
        fig.tight_layout()
    else:
        plt.figure()
        plt.plot(np.real(sig1), label="In-phase(I)")
        plt.plot(np.imag(sig1), label="Quadrature (Q)")
        plt.title("I/Q Time-Domain Plot")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        

