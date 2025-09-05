import numpy as np
import librosa
import scipy.fftpack


def linear_filter_banks(nfilts=20,
                        nfft=512,
                        fs=16000,
                        low_freq=None,
                        high_freq=None,
                        scale="constant"):
    """
    Calculate linear filter banks.
    """
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    linear_points = np.linspace(low_freq, high_freq, nfilts + 2)
    bins = np.floor((nfft + 1) * linear_points / fs).astype(int)
    fbank = np.zeros([nfilts, nfft // 2 + 1])

    if scale in ["descendant", "constant"]:
        c = 1
    else:
        c = 0

    for j in range(nfilts):
        b0, b1, b2 = bins[j], bins[j + 1], bins[j + 2]

        if scale == "descendant":
            c -= 1 / nfilts
            c = max(c, 0)

        elif scale == "ascendant":
            c += 1 / nfilts
            c = min(c, 1)

        fbank[j, b0:b1] = c * (np.arange(b0, b1) - b0) / (b1 - b0)
        fbank[j, b1:b2] = c * (b2 - np.arange(b1, b2)) / (b2 - b1)

    return np.abs(fbank)


def manual_lfcc(
    *,
    y,
    sr: float = 22050,
    n_fft: int,
    hop_length: int,
    n_lfcc: int = 13,
    dct_type: int = 2,
    norm: str = "ortho",
    lifter: float = 0
) -> np.ndarray:
    """
    Compute Linear Frequency Cepstral Coefficients (LFCCs).
    """
    f_b = linear_filter_banks(nfilts=26, nfft=n_fft, fs=sr, scale="descendant")

    X = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    S = librosa.amplitude_to_db(np.abs(X))

    features = np.dot(S.T, f_b.T)
    M: np.ndarray = scipy.fftpack.dct(features.T, axis=-2, type=dct_type, norm=norm)[..., :n_lfcc, :]

    if lifter > 0:
        LI = np.sin(np.pi * np.arange(1, 1 + n_lfcc, dtype=M.dtype) / lifter)
        LI = np.expand_dims(LI, axis=-1)
        M *= 1 + (lifter / 2) * LI
        return M
    elif lifter == 0:
        return M
    else:
        raise ValueError(f"Lifter value {lifter} must be non-negative.")
