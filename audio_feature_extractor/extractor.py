import numpy as np
import pandas as pd
import librosa

from .utils import manual_lfcc


class AudioFeatureExtractor:
    """
    Extracts audio features such as MFCCs, LFCCs, and spectral descriptors.
    """

    def __init__(self, sample_rate: int = 16000, n_fft: int = 2048, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    def _spectral_descriptors(self, signal, descriptor: str) -> np.ndarray:
        """
        Internal method to compute spectral descriptors.

        Parameters:
        - signal (np.ndarray): Audio time series
        - descriptor (str): one of ['centroid','spread','flux','zcr','rms']

        Returns:
        - np.ndarray: Computed feature values
        """
        freq = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft).reshape((-1, 1))
        X = np.abs(librosa.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length, pad_mode='reflect'))

        if descriptor == 'centroid':
            return librosa.feature.spectral_centroid(y=signal, sr=self.sample_rate,
                                                     n_fft=self.n_fft, hop_length=self.hop_length)

        elif descriptor == 'spread':
            cent = librosa.feature.spectral_centroid(y=signal, sr=self.sample_rate,
                                                     n_fft=self.n_fft, hop_length=self.hop_length)
            spread = np.sqrt(np.sum(((freq - cent) ** 2) *
                                    librosa.util.normalize(X, norm=1, axis=0),
                                    axis=0, keepdims=True))
            return spread

        elif descriptor == 'skewness':
            cent = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, n_fft=n_fft, hop_length=512)
            spread = np.sqrt(np.sum(((freq - cent)**2) * librosa.util.normalize(X, norm=1, axis=0), axis=0, keepdims=True))
            skewness = np.sum(((freq - cent)**3) * librosa.util.normalize(X, norm=1, axis=0) / (spread)**3, axis=0, keepdims=True)
            return skewness
    
        elif descriptor == 'kurtosis':
            cent = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, n_fft=n_fft, hop_length=512)
            spread = np.sqrt(np.sum(((freq - cent)**2) * librosa.util.normalize(X, norm=1, axis=0), axis=0, keepdims=True))
            kurtosis = np.sum(((freq - cent)**4) * librosa.util.normalize(X, norm=1, axis=0) / (spread)**4, axis=0, keepdims=True)
            return kurtosis


        elif descriptor == 'flux':
            flux = np.sum(np.abs(np.diff(X, axis=1)), axis=0)
            return flux

         elif descriptor == 'flatness':
             flatness = librosa.feature.spectral_flatness(y=signal, n_fft=self.n_fft, hop_length=self.hop_length)
             return flatness
    
        elif descriptor == 'rolloff':
            rolloff =  librosa.feature.spectral_rolloff(y=signal, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, roll_percent=0.85)
            return rolloff

        elif descriptor == 'crest':
            crest = np.max(X, axis=0) / (np.sum(X, axis=0) / (np.max(freq) - np.min(freq)))
            return crest

        elif descriptor == 'entropy':
            eps = np.finfo(float).eps  # Small epsilon value to avoid log(0) issues
            entropy = -np.sum((X + eps) * np.log(X + eps), axis=0, keepdims=True) / np.log(np.max(freq) - np.min(freq))
            return entropy

        
        elif descriptor == 'zcr':
            return librosa.feature.zero_crossing_rate(y=signal, frame_length=self.n_fft,
                                                      hop_length=self.hop_length)

        elif descriptor == 'rms':
            return librosa.feature.rms(y=signal, frame_length=self.n_fft, hop_length=self.hop_length)

        else:
            raise ValueError(f"Descriptor '{descriptor}' is not recognized.")

    def extract(self, file_name: str, feature_type: str) -> pd.DataFrame:
        """
        Extract features from an audio file.

        Parameters:
        - file_name (str): Path to audio file
        - feature_type (str): 'mfccs', 'lfccs', or 'spectral_shape_descriptors'

        Returns:
        - DataFrame: Extracted features
        """
        signal_audio, sr = librosa.load(file_name, sr=self.sample_rate)
        df = pd.DataFrame()
        n = 0

        if feature_type == 'mfccs':
            MFCCs = librosa.feature.mfcc(y=signal_audio, sr=self.sample_rate, n_fft=self.n_fft,
                                         hop_length=self.hop_length, n_mfcc=13, n_mels=26)
            for m in range(13):
                df.loc[n, f'mfccs_{m}'] = np.mean(MFCCs[m, :])

        elif feature_type == 'lfccs':
            LFCCs = manual_lfcc(y=signal_audio, sr=self.sample_rate, n_fft=self.n_fft,
                                hop_length=self.hop_length, n_lfcc=13)
            for m in range(13):
                df.loc[n, f'lfccs_{m}'] = np.mean(LFCCs[m, :])

        elif feature_type == 'spectral_shape_descriptors':
            descriptors = ['centroid', 'spread', 'flux', 'zcr', 'rms']
            for d in descriptors:
                df.loc[n, d] = self._spectral_descriptors(signal_audio, descriptor=d).mean()

        else:
            raise ValueError(f"Feature type '{feature_type}' is not recognized.")

        return df
