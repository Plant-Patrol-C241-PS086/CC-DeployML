import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image
from io import BytesIO
import json
from flask import Flask, request, jsonify
import base64

app = Flask(__name__)

model = load_model('PlantPatrol_model.h5')

data = [
    {
        "namaTanaman": "Apple Black Root",
        "responTanaman": "Apple__black_rot",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Apple Healthy",
        "responTanaman": "Apple__healthy",
        "obat": "None"
    },
    {
        "namaTanaman": "Apple Rust",
        "responTanaman": "Apple__rust",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Apple Scab",
        "responTanaman": "Apple__scab",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Cassava Bacterial Blight",
        "responTanaman": "Cassava__bacterial_blight",
        "obat": "Copper fungicide"
    },
    {
        "namaTanaman": "Cassava Brown Streak Disease",
        "responTanaman": "Cassava__brown_streak_disease",
        "obat": "Insecticides"
    },
    {
        "namaTanaman": "Cassava Green Mottle",
        "responTanaman": "Cassava__green_mottle",
        "obat": "Insecticides"
    },
    {
        "namaTanaman": "Cassava Healthy",
        "responTanaman": "Cassava__healthy",
        "obat": "None"
    },
    {
        "namaTanaman": "Cassava Mosaic Disease",
        "responTanaman": "Cassava__mosaic_disease",
        "obat": "Insecticides"
    },
    {
        "namaTanaman": "Cherry Healthy",
        "responTanaman": "Cherry__healthy",
        "obat": "None"
    },
    {
        "namaTanaman": "Cherry Powdery Mildew",
        "responTanaman": "Cherry__powdery_mildew",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Chili Healthy",
        "responTanaman": "Chili__healthy",
        "obat": "None"
    },
    {
        "namaTanaman": "Chili Leaf Curl",
        "responTanaman": "'Chili__leaf curl'",
        "obat": "Insecticides, Neem oiltnya"
    },
    {
        "namaTanaman": "Chili Leaf Spot",
        "responTanaman": "'Chili__leaf spot'",
        "obat": "Fungicide, Neem oil"
    },
    {
        "namaTanaman": "Chili Whitefly",
        "responTanaman": "Chili__whitefly",
        "obat": "Insecticidal soaps, Neem oil"
    },
    {
        "namaTanaman": "Chili Yellowish",
        "responTanaman": "Chili__yellowish",
        "obat": "Insecticides"
    },
    {
        "namaTanaman": "Coffee Cercospora Leaf Spot",
        "responTanaman": "Coffee__cercospora_leaf_spot",
        "obat": "Copper ungicides"
    },
    {
        "namaTanaman": "Coffee Healthy",
        "responTanaman": "Coffee__healthy",
        "obat": "None"
    },
    {
        "namaTanaman": "Coffee Red Spider Mite",
        "responTanaman": "Coffee__red_spider_mite",
        "obat": "Acaricides (Miticides)"
    },
    {
        "namaTanaman": "Coffee Rust",
        "responTanaman": "Coffee__rust",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Corn Common Rust",
        "responTanaman": "Corn__common_rust",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Corn Gray Leaf",
        "responTanaman": "Corn__gray_leaf_spot",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Corn Healthy",
        "responTanaman": "Corn__healthy",
        "obat": "None"
    },
    {
        "namaTanaman": "Corn Northern Leaf Blight",
        "responTanaman": "Corn__northern_leaf_blight",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Cucumber Diseased",
        "responTanaman": "Cucumber__diseased",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Cucumber Healthy",
        "responTanaman": "Cucumber__healthy",
        "obat": "None"
    },
    {
        "namaTanaman": "Gauva Diseased",
        "responTanaman": "Gauva__diseased",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Gauva Healthy",
        "responTanaman": "Gauva__healthy",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Grape Black Measles",
        "responTanaman": "Grape__black_measles",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Grape Black Rot",
        "responTanaman": "Grape__black_rot",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Grape Healthy",
        "responTanaman": "Grape__healthy",
        "obat": "None"
    },
    {
        "namaTanaman": "Grape Leaf Blight (isariopsis leaf spot)",
        "responTanaman": "'Grape__leaf_blight_(isariopsis_leaf_spot)'",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Jamun Diseased",
        "responTanaman": "Jamun__diseased",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Jamun Healthy",
        "responTanaman": "Jamun__healthy",
        "obat": "None"
    },
    {
        "namaTanaman": "Lemon Diseased",
        "responTanaman": "Lemon__diseased",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Lemon Healthy",
        "responTanaman": "Lemon__healthy",
        "obat": "None"
    },
    {
        "namaTanaman": "Mango Diseased",
        "responTanaman": "Mango__diseased",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Mango Healthy",
        "responTanaman": "Mango__healthy",
        "obat": "None"
    },
    {
        "namaTanaman": "Peach Bacterial Spot",
        "responTanaman": "Peach__bacterial_spot",
        "obat": "Copper Fungicide"
    },
    {
        "namaTanaman": "Peach Healthy",
        "responTanaman": "Peach__healthy",
        "obat": "None"
    },
    {
        "namaTanaman": "Pepper Bell Bacterial Spot",
        "responTanaman": "Pepper_bell__bacterial_spot",
        "obat": "Copper Fungicide"
    },
    {
        "namaTanaman": "Pepper Bell Healthy",
        "responTanaman": "Pepper_bell__healthy",
        "obat": "None"
    },
    {
        "namaTanaman": "Pomegranate Diseased",
        "responTanaman": "Pomegranate__diseased",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": " Pomegranate Healthy",
        "responTanaman": " Pomegranate__healthy",
        "obat": "None"
    },
    {
        "namaTanaman": "Potato Early Blight",
        "responTanaman": "Potato__early_blight",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Potato Healthy",
        "responTanaman": "Potato__healthy",
        "obat": "None"
    },
    {
        "namaTanaman": "Potato Late Blight",
        "responTanaman": "Potato__late_blight",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Rice Brown Spot",
        "responTanaman": "Rice__brown_spot",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Rice Healthy",
        "responTanaman": "Rice__healthy",
        "obat": "None"
    },
    {
        "namaTanaman": "Rice Hispa",
        "responTanaman": "Rice__hispa",
        "obat": "Insecticides"
    },
    {
        "namaTanaman": "Rice Leaf Blast",
        "responTanaman": "Rice__leaf_blast",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Rice Neck Blast",
        "responTanaman": "Rice__neck_blast",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Soybean Bacterial Blight",
        "responTanaman": "Soybean__bacterial_blight",
        "obat": "Copper Fungicide"
    },
    {
        "namaTanaman": "Soybean Caterpillar",
        "responTanaman": "Soybean__caterpillar",
        "obat": "Insecticides"
    },
    {
        "namaTanaman": "Soybean Diabrotica Speciosa",
        "responTanaman": "Soybean__diabrotica_speciosa",
        "obat": "Insecticides"
    },
    {
        "namaTanaman": "Soybean Downy Mildew",
        "responTanaman": "Soybean__downy_mildew",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Soybean Healthy",
        "responTanaman": "Soybean__healthy",
        "obat": "None"
    },
    {
        "namaTanaman": "Soybean Mosaic Virus",
        "responTanaman": "Soybean__mosaic_virus",
        "obat": "Insecticides"
    },
    {
        "namaTanaman": "Soybean Powdery Mildew",
        "responTanaman": "Soybean__powdery_mildew",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Soybean Rust",
        "responTanaman": "Soybean__rust",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Soybean Southern Blight",
        "responTanaman": "Soybean__southern_blight",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Strawberry Healthy",
        "responTanaman": "Strawberry__healthy",
        "obat": "None"
    },
    {
        "namaTanaman": "Strawberry Leaf Scorch",
        "responTanaman": "Strawberry___leaf_scorch",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Sugarcane Bacterial Blight",
        "responTanaman": "Sugarcane__bacterial_blight",
        "obat": "Copper Fungicide"
    },
    {
        "namaTanaman": "Sugarcane Healthy",
        "responTanaman": "Sugarcane__healthy",
        "obat": "None"
    },
    {
        "namaTanaman": "Sugarcane Red Rot",
        "responTanaman": "Sugarcane__red_rot",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Sugarcane Red Stripe",
        "responTanaman": "Sugarcane__red_stripe",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Sugarcane Rust",
        "responTanaman": "Sugarcane__rust",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Tea Algal Leaf",
        "responTanaman": "Tea__algal_leaf",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Tea Anthracnose",
        "responTanaman": "Tea__anthracnose",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Tea Bird Eye Spot",
        "responTanaman": "Tea__bird_eye_spot",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Tea Brown Blight",
        "responTanaman": "Tea__brown_blight",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Tea Healthy",
        "responTanaman": "Tea__healthy",
        "obat": "None"
    },
    {
        "namaTanaman": "Tea Red Leaf Spot",
        "responTanaman": "Tea__red_leaf_spot",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Tomato Bacterial Spot",
        "responTanaman": "Tomato__bacterial_spot",
        "obat": "Copper Fungicide"
    },
    {
        "namaTanaman": "Tomato Early Blight",
        "responTanaman": "Tomato__early_blight",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Tomato Healthy",
        "responTanaman": "Tomato__healthy",
        "obat": "None"
    },
    {
        "namaTanaman": "Tomato Late Blight",
        "responTanaman": "Tomato__late_blight",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Tomato Leaf Mold",
        "responTanaman": "Tomato__leaf_mold",
        "obat": "Copper Fungicides, Methyl Bromide"
    },
    {
        "namaTanaman": "Tomato Mosaic Virus",
        "responTanaman": "Tomato__mosaic_virus",
        "obat": "Insecticides"
    },
    {
        "namaTanaman": "Tomato Septoria Leaf Spot",
        "responTanaman": "Tomato__septoria_leaf_spot",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Tomato Spider Mites (two spotted spider mite)",
        "responTanaman": "'Tomato__spider_mites_(two_spotted_spider_mite)'",
        "obat": "Acaricides (Amitraz, Abamectin,,Hexythiazox,Spiromesifen,Bifenazate)"
    },
    {
        "namaTanaman": "Tomato Target Spot",
        "responTanaman": "Tomato__target_spot",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Tomato Yellow Leaf Curl Virus",
        "responTanaman": "Tomato__yellow_leaf_curl_virus",
        "obat": "Insecticides"
    },
    {
        "namaTanaman": "Wheat Brown Rust",
        "responTanaman": "Wheat__brown_rust",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": " Wheat Healthy",
        "responTanaman": "Wheat__healthy",
        "obat": "None"
    },
    {
        "namaTanaman": "Wheat Septoria",
        "responTanaman": "Wheat__septoria",
        "obat": "Fungicide"
    },
    {
        "namaTanaman": "Wheat Yellow Rust",
        "responTanaman": "Wheat__yellow_rust",
        "obat": "Fungicide"
    }
]

respon_to_obat = {item["responTanaman"]: item["obat"] for item in data}

classes = {
 0:'Apple__black_rot',
 1:'Apple__healthy',
 2:'Apple__rust',
 3:'Apple__scab',
 4:'Cassava__bacterial_blight',
 5:'Cassava__brown_streak_disease',
 6:'Cassava__green_mottle',
 7:'Cassava__healthy',
 8:'Cassava__mosaic_disease',
 9:'Cherry__healthy',
 10:'Cherry__powdery_mildew',
 11:'Chili__healthy',
 12:'Chili__leaf curl',
 13:'Chili__leaf spot',
 14:'Chili__whitefly',
 15:'Chili__yellowish',
 16:'Coffee__cercospora_leaf_spot',
 17:'Coffee__healthy',
 18:'Coffee__red_spider_mite',
 19:'Coffee__rust',
 20:'Corn__common_rust',
 21:'Corn__gray_leaf_spot',
 22:'Corn__healthy',
 23:'Corn__northern_leaf_blight',
 24:'Cucumber__diseased',
 25:'Cucumber__healthy',
 26:'Gauva__diseased',
 27:'Gauva__healthy',
 28:'Grape__black_measles',
 29:'Grape__black_rot',
 30:'Grape__healthy',
 31:'Grape__leaf_blight_(isariopsis_leaf_spot)',
 32:'Jamun__diseased',
 33:'Jamun__healthy',
 34:'Lemon__diseased',
 35:'Lemon__healthy',
 36:'Mango__diseased',
 37:'Mango__healthy',
 38:'Peach__bacterial_spot',
 39:'Peach__healthy',
 40:'Pepper_bell__bacterial_spot',
 41:'Pepper_bell__healthy',
 42:'Pomegranate__diseased',
 43:'Pomegranate__healthy',
 44:'Potato__early_blight',
 45:'Potato__healthy',
 46:'Potato__late_blight',
 47:'Rice__brown_spot',
 48:'Rice__healthy',
 49:'Rice__hispa',
 50:'Rice__leaf_blast',
 51:'Rice__neck_blast',
 52:'Soybean__bacterial_blight',
 53:'Soybean__caterpillar',
 54:'Soybean__diabrotica_speciosa',
 55:'Soybean__downy_mildew',
 56:'Soybean__healthy',
 57:'Soybean__mosaic_virus',
 58:'Soybean__powdery_mildew',
 59:'Soybean__rust',
 60:'Soybean__southern_blight',
 61:'Strawberry___leaf_scorch',
 62:'Strawberry__healthy',
 63:'Sugarcane__bacterial_blight',
 64:'Sugarcane__healthy',
 65:'Sugarcane__red_rot',
 66:'Sugarcane__red_stripe',
 67:'Sugarcane__rust',
 68:'Tea__algal_leaf',
 69:'Tea__anthracnose',
 70:'Tea__bird_eye_spot',
 71:'Tea__brown_blight',
 72:'Tea__healthy',
 73:'Tea__red_leaf_spot',
 74:'Tomato__bacterial_spot',
 75:'Tomato__early_blight',
 76:'Tomato__healthy',
 77:'Tomato__late_blight',
 78:'Tomato__leaf_mold',
 79:'Tomato__mosaic_virus',
 80:'Tomato__septoria_leaf_spot',
 81:'Tomato__spider_mites_(two_spotted_spider_mite)',
 82:'Tomato__target_spot',
 83:'Tomato__yellow_leaf_curl_virus',
 84:'Wheat__brown_rust',
 85:'Wheat__healthy',
 86:'Wheat__septoria',
 87:'Wheat__yellow_rust'}

# Invert dictionary classes for correct lookup
classes = {v: k for k, v in classes.items()}

# Fungsi untuk melakukan prediksi pada gambar baru dari file upload
def predict_image_from_file(image_file):
    # Baca gambar dari file upload
    image = Image.open(BytesIO(image_file)).convert('RGB')

    # Preprocess gambar
    image = image.resize((224, 224))
    image = img_to_array(image) / 255.0

    # Lakukan prediksi
    prediction = model.predict(np.expand_dims(image, axis=0))[0]
    predicted_label_index = np.argmax(prediction)
    predicted_label = list(classes.keys())[list(classes.values()).index(predicted_label_index)]

    # Dapatkan obat yang sesuai
    obat_predicted = respon_to_obat.get(predicted_label, "Unknown")

    return predicted_label, obat_predicted

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    image_file = file.read()
    predicted_label, obat_predicted = predict_image_from_file(image_file)
    return jsonify({"predicted_label": predicted_label, "obat": obat_predicted})

if __name__ == '__main__':
    app.run(debug=True)