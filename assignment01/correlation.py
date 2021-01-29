import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile


# Question 1: Correlation

def crossCorr(x, y):
    # z[0] corresponds to r_xy(0).
    # This is equivalent to np.correlate(x, y, 'same').
    z = np.zeros(max(len(x), len(y)))
    for eta in range(len(z)):
        length = min(len(x), len(y) - eta)
        z[eta] = np.sum(x[:length] * y[eta:eta+length])
    return z

def loadSoundFile(filename):
    sr, data = wavfile.read(filename)
    # For multichannel files, take the left channel.
    if len(data.shape) > 1:
        data = data[:, 0]
    # Ensure the samples are floats.
    # (Note that this does not scale the range of values.)
    return data.astype(np.float)

def main():
    x = loadSoundFile('snare.wav')
    y = loadSoundFile('drum_loop.wav')
    z = crossCorr(x, y)
    # Plot the result.
    plt.figure(figsize=(16, 8))
    plt.title('Cross-correlation of snare and drum loop')
    plt.xlabel('Sample (eta)')
    plt.ylabel('Cross-correlation value (r_xy(eta))')
    plt.plot(z)
    plt.tight_layout()
    plt.savefig('results/01-correlation.png')


# Question 2: Snare locations

def findSnarePosition(snareFilename, drumloopFilename):
    snare = loadSoundFile(snareFilename)
    drumloop = loadSoundFile(drumloopFilename)
    corr = crossCorr(snare, drumloop)
    # Look for points that are at least 90% of the global maximum.
    pos = signal.find_peaks(corr, height=np.max(corr)*0.9)[0]
    # Return a regular Python list.
    return pos.tolist()


if __name__ == '__main__':
    main()
    with open('results/02-snareLocation.txt', 'w') as out:
        print(findSnarePosition('snare.wav', 'drum_loop.wav'), file=out)
