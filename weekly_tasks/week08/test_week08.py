import numpy as np
import matplotlib.pyplot as plt
import pytest



def test_exercise_1():

    a_vec = [0.9, 0.8, 0.8]
    N_vec = [16, 8, 64]

    for a, N in zip(a_vec,N_vec):

        n = np.arange(0,N)
        k= np.arange(0,N)
        x_n = a**n
        omega = 2*np.pi/N*k
        omega_super_res = np.linspace(0, 2*np.pi,1000)
        X_dtft = 1/(1-a*np.exp(-1j*omega))
        X_dfft_super_res = 1/(1-a*np.exp(-1j*omega_super_res))

        # Generate sampled x
        x = np.fft.ifft(X_dtft)

        # Plot both signals
        fig, ax = plt.subplots(nrows=2, figsize=(10, 8))
        ax[0].set_title(f"Magnitude of Spectra x[n]=a^n*u[n] for a={a}")
        ax[0].plot(omega, np.abs(X_dtft), label = f"DTFT with {N} samples.", color="blue")
        ax[0].plot(omega_super_res,np.abs(X_dfft_super_res), label = f"DTFT with {len(omega_super_res)} samples.", color="black")
        ax[0].set_ylabel("Magnitude")
        ax[0].legend()
        ax[1].set_title("Signal Amplitudes")
        ax[1].plot(n, np.real(x),marker="o",label="x[n] reconstructed with IFFT", color="blue")
        ax[1].plot(n, x_n, marker="o", label="x[n]", color="black")
        ax[1].set_xlabel("Time index")
        ax[1].legend()

    plt.show()