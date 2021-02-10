import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
import time

# Question: If the length of `x` is 200 and the length of `h` is 100, what is the length of `y`?
# Answer: The length of `y` is (200+100-1) = 299
def myTimeConv(x, h):
    y = np.zeros(len(x) + len(h) - 1)
    h = h[::-1]  # flip h
    for n in range(len(y)):
        m_start = max(0, n-len(h)+1)
        m_end = min(len(x), n+1)
        h_offset = len(h) - 1 - n  # account for flipped indices
        y[n] = np.sum(x[m_start:m_end] * h[h_offset+m_start:h_offset+m_end])
    return y


def CompareConv(x, h):
    start = time.time()
    a = myTimeConv(x, h)
    mid = time.time()
    b = signal.convolve(x, h)
    end = time.time()
    diff = a - b
    m = np.mean(diff)
    mabs = np.abs(m)
    stddev = np.std(diff)
    times = np.array([mid - start, end - mid])
    return (m, mabs, stddev, times)


def main():
    # Setup the inputs
    x = np.ones(200)
    h = np.zeros(51)
    h[:26] = np.linspace(0, 1, 26)
    h[26:] = h[24::-1]
    # Convolve
    y_time = myTimeConv(x, h)
    # Plot the result
    plt.figure(figsize=(16, 8))
    plt.title('Convolution of x and h')
    plt.xlabel('Sample [n]')
    plt.ylabel('Convolution value (x * h)[n]')
    plt.plot(y_time)
    plt.tight_layout()
    plt.savefig('results/01-convolution.png')

    x = wavfile.read('audio/piano.wav')[1].astype(np.float)
    h = wavfile.read('audio/impulse-response.wav')[1].astype(np.float)
    m, mabs, stddev, times = CompareConv(x, h)
    with open('results/02-comparison.txt', 'w') as out:
        print(f"m = {m}",
              f"mabs = {mabs}",
              f"stddev = {stddev}",
              f"myTimeConv time: {times[0]} s",
              f"scipy.signal.convolve time: {times[1]} s",
              sep='\n', file=out)


if __name__ == '__main__':
    main()