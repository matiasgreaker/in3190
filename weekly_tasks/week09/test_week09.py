import matplotlib.pyplot as plt
import numpy as np

def series(n):
    """Generate the n-th element"""
    return n*((0.9)**n)

def dtft_of_series(radians):
    """Get the DTFT based on the analytic formula"""
    val = 0.9*np.exp(-np.complex128(0, radians))
    return (val)/((1-val)**2)

def frequency_in_radians(n,N):
    """Get the frequencies in radians"""
    return 2*np.pi*n/N


def test_exercise_2():
    # Vectorize functions
    vectorized_series = np.vectorize(series)
    vectorized_freq = np.vectorize(frequency_in_radians, excluded='N')
    dtft = np.vectorize(dtft_of_series)

    # Do the N DFT
    N = 20
    n = np.arange(0, N)
    x_n = vectorized_series(n)
    X_DTFT_N = np.fft.fft(x_n)
    freq = vectorized_freq(n, N)

    # Do the super accurate DTFT
    radians = np.linspace(-np.pi, np.pi, 1000)
    X_DTFT = dtft(radians)

    # D0 the same for 50 and 100 points
    N = 50
    n = np.arange(0, N)
    x_n = vectorized_series(n)
    X_DTFT_50 = np.fft.fft(x_n)
    freq_50 = vectorized_freq(n, N)

    N = 100
    n = np.arange(0, N)
    x_n = vectorized_series(n)
    X_DTFT_100 = np.fft.fft(x_n)
    freq_100 = vectorized_freq(n, N)

    fig, ax = plt.subplots(nrows=3,sharex=True)
    ax[0].stem(freq-np.pi, np.log10(np.abs(np.fft.fftshift(X_DTFT_N))))
    ax[0].plot(radians, np.log10(np.abs(X_DTFT)), color="g")
    ax[0].set_title("DFT of 20 points + DTFT.")

    ax[1].stem(freq_50-np.pi, np.log10(np.abs(np.fft.fftshift(X_DTFT_50))))
    ax[1].plot(radians, np.log10(np.abs(X_DTFT)), color="g")
    ax[1].set_title("DFT of 50 points + DTFT.")

    ax[2].stem(freq_100-np.pi, np.log10(np.abs(np.fft.fftshift(X_DTFT_100))))
    ax[2].plot(radians, np.log10(np.abs(X_DTFT)), color="g")
    ax[2].set_title("DFT of 100 points + DTFT.")
    ax[2].set_xlabel("Rad/sample")

    plt.show()