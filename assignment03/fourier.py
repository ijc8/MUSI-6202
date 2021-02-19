import matplotlib.pyplot as plt
import numpy as np
import os


def generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):
    length_samples = round(length_secs * sampling_rate_Hz)
    t = np.arange(length_samples) / sampling_rate_Hz
    x = amplitude * np.sin(2*np.pi*frequency_Hz*t + phase_radians)
    return (t, x)


def generateSquare(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):
    # Sum first 10 odd-integer harmonics.
    x = 4/np.pi * np.sum([generateSinusoidal(amplitude, sampling_rate_Hz, (2*k + 1) * frequency_Hz, length_secs, phase_radians)[1]/(2*k + 1) for k in range(10)], axis=0)
    t = np.arange(len(x)) / sampling_rate_Hz
    return (t, x)


def computeSpectrum(x, sample_rate_Hz):
    # NOTE: The non-redundant part includes the Nyquist bin,
    # so the length here will be len(x)/2 + 1, when len(x) is even.
    X = np.fft.fft(x)[:len(x)//2+1]
    f = np.linspace(0, sample_rate_Hz, len(x), endpoint=False)[:len(x)//2+1]
    XAbs = np.abs(X)
    XPhase = np.angle(X)
    XRe = X.real
    XIm = X.imag
    return (f, XAbs, XPhase, XRe, XIm)


def generateBlocks(x, sample_rate_Hz, block_size, hop_size):
    # This is the minimum N necessary to ensure every value in x is included in X.
    N = int(np.ceil((len(x) - block_size) / hop_size)) + 1
    len(x) - block_size
    X = np.zeros((block_size, N))
    for i in range(N):
        X[:min(len(x) - hop_size*i, block_size), i] = x[hop_size*i:hop_size*i + block_size]
    t = np.arange(N) * hop_size / sample_rate_Hz
    return (t, X)


def mySpecgram(x, block_size, hop_size, sampling_rate_Hz, window_type):
    window = {'rect': np.ones, 'hann': np.hanning}[window_type](block_size)
    t, blocks = generateBlocks(x, sampling_rate_Hz, block_size, hop_size)
    time_vector = t.reshape((-1, 1))
    fft_blocks = []
    for i, block in enumerate(blocks.T):
        # NOTE: For reasons described in computeSpectrum(), the length of the spectrum may be block_size/2 + 1, not block_size/2.
        # In the interests of including all non-redunant information, I keep the extra element, which affects the shape of freq_vector and magnitude_spectrum.
        f, XAbs, *_ = computeSpectrum(block * window, sampling_rate_Hz)
        fft_blocks.append(XAbs)
    freq_vector = f.reshape((-1, 1))
    magnitude_spectrogram = np.array(fft_blocks).T
    plt.figure(figsize=(16, 8))
    plt.title(f'Square Wave Spectrogram ({window_type.capitalize()} window)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt_spectrum, plt_freqs, plt_times, im = plt.specgram(
        # Pad to ensure that this includes the last block.
        np.concatenate((x, np.zeros((len(t) - 1) * hop_size + block_size - len(x)))),
        NFFT=block_size,
        noverlap=block_size-hop_size,
        Fs=sampling_rate_Hz,
        window=lambda a: a*window,
        mode='magnitude',
        scale_by_freq=False
    )
    plt.colorbar(im, label='Magnitude (dB)')
    # Compare our results.
    assert(np.isclose(freq_vector[:, 0], plt_freqs).all())
    assert(np.isclose(magnitude_spectrogram / np.sum(window), plt_spectrum).all())
    # NOTE: Can't modify function signature, so rename this output manually after running.
    plt.savefig('results/specgram.png')
    return (freq_vector, time_vector, magnitude_spectrogram)


def sineSweep(start_freq, end_freq, duration, sample_rate):
    out = np.zeros(int(duration * sample_rate))
    freqs = np.linspace(start_freq, end_freq, len(out))
    phases = np.cumsum(freqs)*2*np.pi/sample_rate
    sweep = np.sin(phases)
    return computeSpectrum(sweep, sample_rate)


def main():
    sample_rate = 44100

    sin_t, sin_x = generateSinusoidal(1.0, sample_rate, 400, 0.5, np.pi/2)
    # Plot the result
    plt.figure(figsize=(16, 8))
    plt.title('Sinusoidal')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.plot(sin_t[:int(5e-3 * sample_rate)], sin_x[:int(5e-3 * sample_rate)])
    plt.tight_layout()
    plt.savefig('results/01-sinusoid.png')

    sqr_t, sqr_x = generateSquare(1.0, sample_rate, 400, 0.5, 0)
    # Plot the result
    plt.figure(figsize=(16, 8))
    plt.title('Square wave')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.plot(sqr_t[:int(5e-3 * sample_rate)], sqr_x[:int(5e-3 * sample_rate)])
    plt.tight_layout()
    plt.savefig('results/02-square.png')

    # Plot spectrums
    for (i, name, t, x) in [(3, 'sinusoid', sin_t, sin_x), (4, 'square', sqr_t, sqr_x)]:
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle(f'{name.capitalize()} Spectrum')
        top, bottom = fig.subplots(2, sharex=True)
        f, XAbs, XPhase, XRe, XIm = computeSpectrum(x, sample_rate)
        top.plot(f, XAbs)
        top.set_ylabel('Magnitude')
        bottom.plot(f, XPhase)
        bottom.set_xlabel('Frequency (Hz)')
        bottom.set_ylabel('Phase (rad)')
        plt.tight_layout()
        plt.savefig(f'results/{i:02}-{name}-spectrum.png')
    
    with open('results/05-resolution.txt', 'w') as out:
        print(f'Frequency resolution: {f[1] - f[0]} Hz', file=out)
        print(f'Frequency resolution with zero-padding: {(f[1] - f[0])/2} Hz', file=out)

    mySpecgram(sqr_x, 2048, 1024, sample_rate, 'rect')
    os.rename('results/specgram.png', 'results/06-rect-specgram.png')
    mySpecgram(sqr_x, 2048, 1024, sample_rate, 'hann')
    os.rename('results/specgram.png', 'results/07-hann-specgram.png')

    f, XAbs, XPhase, *_ = sineSweep(0, 22050, 1.0, sample_rate)
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('Sine-Sweep Spectrum')
    top, bottom = fig.subplots(2, sharex=True)
    top.plot(f, XAbs)
    top.set_ylabel('Magnitude')
    bottom.plot(f, XPhase)
    bottom.set_xlabel('Frequency (Hz)')
    bottom.set_ylabel('Phase (rad)')
    plt.tight_layout()
    plt.savefig(f'results/09-sine-sweep-spectrum.png')


if __name__ == '__main__':
    main()