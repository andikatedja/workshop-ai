from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('model.h5')

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Kalau user memasukkan nilai dan submit
    if request.method == 'POST':
        # Fetch data
        x1 = request.form.get('sepalLength')
        x2 = request.form.get('sepalWidth')
        x3 = request.form.get('petalLength')
        x4 = request.form.get('petalWidth')

        # Normalisasi data
        xMean = np.array([5.84333333, 3.05733333, 3.758, 1.19933333])
        xStd = np.array([0.82530129, 0.43441097, 1.75940407, 0.75969263])

        x1_norm = (np.float(x1) - xMean[0]) / xStd[0]
        x2_norm = (np.float(x2) - xMean[1]) / xStd[1]
        x3_norm = (np.float(x3) - xMean[2]) / xStd[2]
        x4_norm = (np.float(x4) - xMean[3]) / xStd[3]

        # Combine data ke numpy array 2d
        feature = np.array([[x1_norm, x2_norm, x3_norm, x4_norm]])

        # Lakukan prediksi
        hasil = model.predict(feature)

        # Cari index terbesar hasil prediksi/cari kelasnya
        kelas = np.argmax(hasil, axis=1)

        nama_kelas = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
        hasil_prediksi = nama_kelas[int(kelas)]
        return render_template('predict.html', prediction_text='{}'.format(hasil_prediksi), img='{}'.format(hasil_prediksi + '.jpg'))
    else:
        return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)