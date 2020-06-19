# import
import numpy as np
import matplotlib.pyplot as plt

# Adaptive Filter: NLMS algorithm
def nlms(d, x, n, mu):
    # inputs
    #   d: Desired response
    #   x: Input data
    #   n: Filter length
    #   mu: step size
    # output
    #   s: output data(signal)

    # -------------------------------------
    # init
    # -------------------------------------
    # phi: The vector of buffered input data at step i
    phi = np.zeros(n)
    # filter weight
    w = np.zeros(n)
    # output data
    s = np.zeros(len(x))
    
    # Filterring
    for i, x_i in enumerate(x):
        # buffering
        phi[1:] = phi[0:-1]
        phi[0] = x_i

        # e: error
        e = d[i] - np.dot(w, phi)
        
        # filter update
        w = w + mu * e / (0.01 + np.dot(phi, phi)) * phi

        # output
        s[i] = e

    return s

def main():

    # -------------------------------------
    # parameter setting
    # -------------------------------------
    # time max
    time_max = 1
    # time step (sampling rate)
    dt = 0.001

    # -------------------------------------
    # create sample data
    # -------------------------------------
    # time step
    time = np.arange(0, time_max, dt)
    # noise
    noise = 0.3 * np.random.randn(len(time))

    # lpf
    from scipy import signal
    fn = 1 / (dt * 2)
    N, Wn = signal.buttord(300 / fn, 500 / fn, 1, 40)
    b, a = signal.butter(N, Wn, "low")
    noise_filtered = signal.filtfilt(b, a, noise)

    # sample data
    data = np.sin(2*np.pi*5*time) + noise_filtered

    # -------------------------------------
    # adaptive filter
    # -------------------------------------
    # denoising
    signal = nlms(data, noise, 64, 0.1)

    # **********************************************
    # plot
    # **********************************************
    plt.plot(data)
    plt.plot(signal)
    plt.show()

if __name__ == '__main__':
    main()

