from audio_feature_extractor.extractor import AudioFeatureExtractor
import numpy as np

if __name__ == "__main__":
    # Initialize the extractor
    extractor = AudioFeatureExtractor(sample_rate=16000)

    # Extract MFCC features as a NumPy array
    features = extractor.extract("example.wav", feature_type="mfccs")

    # Print the array
    print("Extracted features (NumPy array):")
    print(features)

    # Optionally, print shape
    print("Feature shape:", features.shape)
