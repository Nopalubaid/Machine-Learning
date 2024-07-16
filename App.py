import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
import pickle
import os
import pandas as pd


@st.cache_resource
def load_model_and_tokenizer(model_path, tokenizer_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None, None
    if not os.path.exists(tokenizer_path):
        st.error(f"Tokenizer file not found: {tokenizer_path}")
        return None, None

    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model_path = 'spam_detection_cnn_model.h5'
tokenizer_path = 'tokenizer.pickle'

model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)

if model is None or tokenizer is None:
    st.stop()

def predict_spam(text, model, tokenizer, maxlen=100):
    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_seq, padding='post', maxlen=maxlen)
    prediction = (model.predict(text_pad) > 0.5).astype("int32")
    return "Scam" if prediction[0][0] == 1 else "Normal"

st.title('Scam Detection App')
st.write("""
Aplikasi ini menggunakan model pembelajaran mesin untuk mendeteksi apakah pesan teks yang dimasukkan adalah scam atau tidak. 
Masukkan pesan Anda di bawah ini dan klik tombol 'Predict' untuk melihat hasilnya.
""")
st.write("Contoh pesan yang bisa Anda coba: 'You have won a lottery! Claim your prize now.'")

user_input = st.text_area("Masukkan pesan:")

uploaded_file = st.file_uploader("upload gambar yang mengandung pesan:", type=['txt'])
if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    st.text_area("File content:", content)
    if st.button('Predict from File'):
        if content:
            results = [predict_spam(line, model, tokenizer) for line in content.split('\n') if line.strip()]
            for i, line in enumerate(content.split('\n')):
                if line.strip():
                    st.write(f"Message: {line} - Prediction: {results[i]}")
        else:
            st.write("The uploaded file is empty.")

if st.button('Predict'):
    if user_input:
        with st.spinner('Predicting...'):
            result = predict_spam(user_input, model, tokenizer)
        st.success(f"Prediction: {result}")
    else:
        st.write("Masukkan pesan terlebih dahulu.")

results = {'Scam': 10, 'Normal': 90}  # Contoh data hasil prediksi
results_df = pd.DataFrame(list(results.items()), columns=['Category', 'Count'])

# st.bar_chart(results_df.set_index('Category'))

st.sidebar.title("Informasi Tambahan")
st.sidebar.write("Aplikasi ini dibangun menggunakan Streamlit dan Keras. Untuk informasi lebih lanjut, kunjungi [GitHub](https://github.com/).")

st.sidebar.title("Kontak")
st.sidebar.write("""
Jika Anda memiliki pertanyaan atau memerlukan bantuan, silakan hubungi:
- Email: support@scamdetector.com
- Telepon: +62-123-456-789
""")
