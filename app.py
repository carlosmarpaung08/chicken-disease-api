from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from flask_cors import CORS

# Inisialisasi aplikasi Flask
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Load model klasifikasi 4 kelas
model = load_model('model/chicken_disease_model_final.h5')

# Daftar nama kelas sesuai urutan saat training
class_labels = [
    'Chicken_Coccidiosis',
    'Chicken_Healthy',
    'Chicken_NewCastleDisease',
    'Chicken_Salmonella'
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        try:
            # Load dan preprocess gambar
            img = image.load_img(BytesIO(file.read()), target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Prediksi
            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))

            predicted_class = class_labels[predicted_index]

            return jsonify({
                'prediction': predicted_class,
                'confidence': round(confidence, 4)
            })
        
        except Exception as e:
            print(f"Error: {e}")
            return jsonify({'error': 'Failed to process image'})
    
    return jsonify({'error': 'Invalid file'})

if __name__ == "__main__":
    app.run(debug=True)
