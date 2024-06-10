import librosa 
import numpy as np 
from feature_extraction import extract_features


def get_predict_feat(path, scaler):
    d, s_rate = librosa.load(path, duration=2.5, offset=0.6)
    res = extract_features(d, sr=s_rate)
    result = np.array(res)
    if result.size != 1620:
        raise ValueError(f"Unexpected feature size: {result.size}, expected: 1620")
    result = np.reshape(result, newshape=(1, 1620))
    i_result = scaler.transform(result)
    final_result = np.expand_dims(i_result, axis=2)
    return final_result


def prediction(audio_path, model, encoder, scaler):
    res=get_predict_feat(audio_path, scaler)
    predictions=model.predict(res)
    y_pred = encoder.inverse_transform(predictions)
    print(y_pred[0][0])   