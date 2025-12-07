#!/usr/bin/env python3
import sys
import os
import matplotlib.pyplot as plt
import numpy as np


#Local imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)
import plottingLib

plt.style.use('dark_background')


spec = np.loadtxt("spectrum.csv")

plt.figure(figsize=(6, 4))
x = np.linspace(-len(spec), len(spec), len(spec), endpoint=False)
plt.plot(x, spec)
plt.title('FFT Magnitude of IQ Data')
plt.xlabel('bin')
plt.ylabel('Magnitude')

plt.show()