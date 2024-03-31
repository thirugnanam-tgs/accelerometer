import joblib, os
import streamlit as st
from tqdm import tqdm
import pandas as pd, numpy as np

from statistics import mode
from scipy.signal import spectrogram
from scipy.stats import skew, kurtosis


model = joblib.load(os.path.join(r'artifacts\model_all_data', 'model.pkl'))



def segment_signal(signal, window_length_sec=1, overlap_ratio=0.5, sampling_rate=52):
    window_length_samples = int(window_length_sec * sampling_rate)
    overlap_length = int(window_length_samples * overlap_ratio)
    stride = window_length_samples - overlap_length
    
    segments = []
    for i in range(0, len(signal), stride):
        segment = signal[i:i + window_length_samples]
        segments.append(segment)
    
    return segments



def extract_features(segment_data, sampling_rate = 52):
    features = [np.mean(segment_data[['x','y','z']].apply(lambda df_: np.sqrt(df_.x**2 + df_.y**2 + df_.z**2), axis=1))]
    features.append(np.mean(np.correlate(segment_data['x'], segment_data['y'], mode='full')))
    features.append(np.mean(np.correlate(segment_data['x'], segment_data['z'], mode='full')))
    features.append(np.mean(np.correlate(segment_data['z'], segment_data['y'], mode='full')))
    
    for col in ['x', 'y', 'z']:
        segment = segment_data[col]
        features.append(np.mean(segment))
        features.append(np.var(segment))
        features.append(np.std(segment))
        features.append(skew(segment))
        features.append(kurtosis(segment))
        features.append(np.percentile(segment, 25))
        features.append(np.percentile(segment, 50))
        features.append(np.percentile(segment, 75))
        features.append(np.min(segment))
        features.append(np.max(segment))

    segment = segment_data[['xf', 'yf', 'zf']]
    _, _, Sxx = spectrogram(segment, fs=sampling_rate, nperseg=3)
    mean_power = np.mean(Sxx)
    max_power = np.max(Sxx)
    min_power = np.min(Sxx)

    features.append(mean_power)
    features.append(max_power)
    features.append(min_power)
    
    return features, segment_data.index



def predict_outcome(data):
    
    data['xf'] = np.real(np.fft.fft(data['x']))
    data['yf'] = np.real(np.fft.fft(data['y']))
    data['zf'] = np.real(np.fft.fft(data['z']))

    segments = segment_signal(signal=data)

    train_data = [extract_features(segment) for segment in tqdm(segments, total=len(segments))]
    X = pd.DataFrame([i for i, _ in train_data])
    indices = pd.Series([idx for _, idx in train_data])

    
    predictions = model.predict(X)
    print(len(predictions), len(indices))
    for pred, idxs in zip(predictions, indices):
        data.loc[idxs, 'outcome'] = pred

    return data



def main():
    st.title('CSV File Upload and Prediction App')
    
    st.sidebar.title('Upload CSV File')
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=None)
        df.columns = ['slno', 'x', 'y', 'z', 'label']

        st.subheader('Uploaded Data')
        st.write(df)
        
        predictions = predict_outcome(df)
        
        st.subheader('Predicted Outcomes')
        st.write(predictions)



if __name__ == '__main__':
    main()
