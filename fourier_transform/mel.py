import numpy as np
from typing import Union

def convert_hz_to_mel(f: Union[np.float64, np.ndarray]) -> Union[np.float64, np.ndarray]:
    """
    Converts a frequency in Hz to Mels

    Arguments:
        - f (np.float64 or np.ndarray): frequency in Hz
    Returns:
        - m (np.float64 or np.ndarray): frequency in Mels
    """
    return 2592 * np.log10(1 + (f / 700))

def convert_mel_to_hz(m: Union[np.float64, np.ndarray]) -> Union[np.float64, np.ndarray]:
    """
    Converts a frequency in Mels to Hz

    Arguments:
        - m (np.float64 or np.ndarray): frequency in Mels
    Returns:
        - f (np.float64 or np.ndarray): frequency in Hz
    """
    return 700 * (np.power(10, m / 2592) - 1)

def create_mel_filterbanks(num_mel_bands: np.int16, frame_size: np.int32, sr: np.int32) -> np.ndarray:
    """
    Applys Mel filterbanks to a spectrogram to scale the frequencies in a logarithmic manner

    Parameters
    ----------
        num_mel_bands : np.int16
            The number of mel bands to use
        frame_size : np.int32
            The number of samples per frame in the STFT
        sr : np.int32
            The sample rate of the original signal

    Returns
    -------
        M : np.ndarray [shape=(num_mel_bands, (frame_size / 2) + 1)]
            A linear transformation matrix to scale a vanilla spectrogram
    """
    # Convert the lowest and highest frequencies of the spectrogram to Mels
    lowest_freq = convert_hz_to_mel(0)
    highest_freq = convert_hz_to_mel(sr / 2) # Nyquist frequency

    # Get (mel_bands) evenly spaced points between the lowest and highest Mel frequencies
    mel_centers = np.linspace(start=lowest_freq, stop=highest_freq, num=num_mel_bands+2)

    # Convert these points back to Hz
    mel_centers_hz = convert_mel_to_hz(mel_centers)

    # Get the center frequencies of each FFT bin
    fft_bins = np.linspace(0, sr // 2, (frame_size // 2) + 1)

    # Initialize the Mel filter bank matrix
    M = np.zeros(shape=(num_mel_bands, (frame_size // 2) + 1))

    # Get the differences between all of the mel centers
    mel_band_widths = np.diff(mel_centers_hz)

    # Get the differences between each of the mel centers and all of the FFT bin centers
    mel_to_fft_diffs = np.subtract.outer(mel_centers_hz, fft_bins)

    # Create the filter bank
    for i in range(0, num_mel_bands):
        # Calculate the filter slopes
        rising_slope = -mel_to_fft_diffs[i] / mel_band_widths[i]
        falling_slope = mel_to_fft_diffs[i + 2] / mel_band_widths[i + 1]

        # Intersect the slopes to create the triangular filters
        M[i] = np.maximum(0, np.minimum(rising_slope, falling_slope))

    # This is the only part I don't quite understand, I'll figure it out later
    enorm = 2.0 / (mel_centers_hz[2 : num_mel_bands + 2] - mel_centers_hz[:num_mel_bands])
    M *= enorm[:, np.newaxis]

    return M