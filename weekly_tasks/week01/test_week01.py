import os
import pytest
import numpy as np
import matplotlib.pyplot as plt

def test_exercise_1():
    t = np.linspace(-1, 2.5, 1000)
    f1 = np.cos(2*np.pi*t)
    f2 = np.cos(2*np.pi*t+np.pi)
    f3 = np.cos(8*np.pi*t)
    f4 = np.cos(4*np.pi*t-np.pi/3)

    fig, ax = plt.subplots()
    ax.plot(t, f1, label='f1: cos(2πt)')
    ax.plot(t, f2, label='f2: cos(2πt+π)')
    ax.plot(t, f3, label='f3: cos(8πt)')
    ax.plot(t, f4, label='f4: cos(4πt-π/3)')
    ax.legend()
    ax.set_xlabel('t')
    ax.set_title("Exercise 1: Cosine Functions")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(current_dir, 'exercise_1_cosine_functions.png'))
