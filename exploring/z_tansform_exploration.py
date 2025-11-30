from scipy import signal
import numpy as np
import matplotlib.pyplot as plt





def evaluate_transfer_function(a: np.ndarray, b: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Evaluate the transfer function H(z) = B(z)/A(z) at the points in z.
    Assumes the system is of the form H(z=B(z)/A(z))=b[0] + b[1]z^-1 + ... + b[M]z^-M / a[0] + a[1]z^-1 + ... + a[N]z^-N
    Returns an array of the same shape as z with the evaluated values.
    """
    # Evaluate numerator
    B = np.zeros_like(z, dtype=np.complex128)
    for i, coeff in enumerate(b):
        B += coeff * z**(len(b)-i-1)
    # Evaluate denominator
    A = np.zeros_like(z, dtype=np.complex128)
    for i, coeff in enumerate(a):
        A += coeff * z**(len(a)-i-1)
    return B / A


def plot_zplane(a: np.ndarray, b: np.ndarray, num_points: int = 500, zlim: bool = True) -> None:
    """Take the numerator and denominator coefficients of a system and plot its z-plane.
    Assumes the system is of the form H(z=B(z)/A(z))=b[0] + b[1]z^-1 + ... + b[M]z^-M / a[0] + a[1]z^-1 + ... + a[N]z^-N
    Parameters:
    a: Denominator coefficients of the transfer function.
    b: Numerator coefficients of the transfer function.
    num_points: Number of points to use in each dimension for the meshgrid.
    zlim: Whether to limit the z-axis to avoid distortion from poles.
    """
    z, p, k = signal.tf2zpk(b, a)
    # Make sure we generate a big enough meshgrid
    # This is the radius of the outermost pole/zero
    # Plot at least 25% larger than that
    r_max = np.max(np.abs(np.concatenate((z, p))))*1.25
    # Create a grid of points in the complex plane
    real = np.linspace(-r_max, r_max, num_points)
    imag = np.linspace(-r_max, r_max, num_points)
    X, Y = np.meshgrid(real, imag)
    Z = X + 1j * Y

    # Evaluate the transfer function on the grid
    H = evaluate_transfer_function(a, b, Z)

    # Also create a unit circle for reference
    theta = np.linspace(0, 2 * np.pi, num_points)
    unit_circle = np.exp(1j * theta)


    # Calculate the impulse response
    n, impulse_response = signal.dimpulse((b, a, 1), n=100)
    print(len(n))
    impulse_response = np.reshape(impulse_response, shape=(-1,1))
    # Plot the 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(122, projection='3d')
    ax.plot_surface(X, Y, np.abs(H), cmap='viridis')
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_zlabel('|H(z)|')
    # The poles often trow the scale of the plot off, hence we limit the z-axis to less than the pole values
    if zlim:
        zlimit = np.max(np.abs(H))*0.05
        ax.set_zlim(0, zlimit)
    ax.set_title('Z-Plane')
    # Plot the unit circle and poles/zeros
    ax2 = fig.add_subplot(221)
    ax2.plot(np.real(unit_circle), np.imag(unit_circle),  linestyle="-", color='black')
    ax2.plot(np.real(z), np.imag(z), 'o', label='Zeros', markersize=10)
    ax2.plot(np.real(p), np.imag(p), 'x', label='Poles', markersize=10)
    ax2.set_xlabel('Real Part')
    ax2.set_ylabel('Imaginary Part')
    ax2.set_title('Unit circle with poles and zeros')
    ax2.legend()
    # Plot the impulse response
    ax3 = fig.add_subplot(223)
    ax3.stem(n, impulse_response)
    ax3.set_xlabel('n (samples)')
    ax3.set_ylabel('Impulse Response h[n]')
    ax3.set_title('Impulse Response')

    plt.grid()
    plt.show()

b = np.array([1])
a = np.array([1,0,0,0,0,0.9])
plot_zplane(a, b)