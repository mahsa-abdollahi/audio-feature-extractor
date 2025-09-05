from audio_feature_extractor.extractor import AudioFeatureExtractor

if __name__ == "__main__":
    # Initialize the extractor
    extractor = AudioFeatureExtractor(sample_rate=16000)

    # Extract MFCCs
    mfcc_features = extractor.extract("example.wav", feature_type="mfccs")
    print("MFCC features:", mfcc_features)

    # Extract LFCCs
    lfcc_features = extractor.extract("example.wav", feature_type="lfccs")
    print("LFCC features:", lfcc_features)

    # Extract spectral shape descriptors
    spectral_features = extractor.extract("example.wav", feature_type="spectral_shape_descriptors")
    print("Spectral descriptors:", spectral_features)
