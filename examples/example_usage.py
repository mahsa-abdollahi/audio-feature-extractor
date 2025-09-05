from audio_feature_extractor.extractor import AudioFeatureExtractor

if __name__ == "__main__":
    extractor = AudioFeatureExtractor(sample_rate=16000)
    features = extractor.extract("example.wav", feature_type="mfccs")
    print(features)
