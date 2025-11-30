import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def test_exercise_4():
    """Doing exercise 2 week 12"""
    # passband_edge 2
    omega_p = 2
    # stopband edge
    omega_s = 4
    # Passband ripple
    A_p = 2
    # Stopband att.
    A_s = 30


    # a)
    # Finn chebichev filter order N and response H(s) of analog filter
    N, Wn = signal.cheb1ord(wp=omega_p, ws=omega_s, gpass=A_p, gstop=A_s, analog=True, fs=None)
    b, a = signal.cheby1(N, A_p, Wn, 'low', analog=True)

    # b)
    # Find the digital filter
    b_d, a_d = signal.bilinear(b, a)

    # c)
    omegas = np.linspace(0, 7, 1000)
    # Create the analog transfer function
    H_c_list = []
    num_N = len(b)
    den_N = len(a)
    for i, omega in enumerate(omegas):
        num = 0
        den = 0
        for i, b_i in enumerate(b):
            num = num+b_i*(1j*omega)**(num_N-1-i)
        for i, a_i in enumerate(a):
            den = den+a_i*(1j*omega)**(den_N-1-i)
        H_c_list.append(num/den)

    # Create the digital transfer function
    H_d_list = []
    for i, omega in enumerate(omegas):
        num = 0
        den = 0
        for i, b_i in enumerate(b_d):
            num = num+b_i*np.exp(-1j*omega*i)
        for i, a_i in enumerate(a_d):
            den = den+a_i*np.exp(-1j*omega*i)
        H_d_list.append(num/den)

    # Plot analog and digital
    H_c = np.array(H_c_list)
    H_d = np.array(H_d_list)
    fig, ax = plt.subplots()
    ax.plot(omegas, 20 * np.log10(abs(H_c)), label="My analog filter.")
    ax.plot(omegas, 20 * np.log10(abs(H_d)), label="My digital filter.")
    ax.legend()
    ax.set_title(f"Chebyshev Type I frequency response (N={N})")
    ax.set_xlabel('Frequency [rad/s]')
    ax.set_ylabel('Amplitude [dB]')
    ax.margins(0, 0.1)
    ax.grid(which='both', axis='both')
    ax.axvline(omega_p, color='green') # cutoff frequency
    ax.axhline(-A_p, color='green') # rp

    # d) Verify the plots
    w, h = signal.freqs(b, a, worN=omegas)
    w_d, h_d = signal.freqz(b_d,a_d,worN=omegas,include_nyquist=True)

    #fig, ax = plt.subplots()
    ax.semilogx(w, 20 * np.log10(abs(h)), label ="Using freqs (analog) to plot", ls='dashed')
    ax.semilogx(w_d, 20 * np.log10(abs(h_d)), label = "Using freqz (digital) to plot",ls='dashed')
    ax.legend()
    plt.show()
