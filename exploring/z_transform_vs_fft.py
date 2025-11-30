import numpy as np

h = np.array([1/3, 1/3, 1/3])


# Take fourier Transform of h
H_fft = np.fft.fft(h, n=512)
freqs = np.fft.fftfreq(512)
freqs = np.fft.fftshift(freqs)
H_fft = np.fft.fftshift(H_fft)

# Take Z-transform of h at z = exp(j*2*pi*f)
f = np.linspace(-0.5, 0.5, 512, endpoint=False)
z = np.exp(1j * 2 * np.pi * f)
H_z = 1/3 + 1/3 * z**-1 + 1/3 * z**-2

# Compare the results
import matplotlib.pyplot as plt
fig, axs = plt.subplots()
axs.plot(freqs, np.abs(H_fft), label='FFT Magnitude')
axs.plot(f, np.abs(H_z), '--', label='Z-transform Magnitude')
axs.set_xlabel('Frequency (cycles/sample)')
axs.set_ylabel('Magnitude')
axs.legend()
plt.show()