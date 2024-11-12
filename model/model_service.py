from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Chargement du modèle et du scaler
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Chargement des données pour les valeurs médianes
df = pd.read_csv("cleaned_data.csv", sep=',', on_bad_lines='skip', low_memory=False)

def parse_input(value, nom):
    try:
        if value is None or value == "" or value == "NaN":
            return df[nom].median()
        return float(value)
    except (ValueError, KeyError):
        return 0.0

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validation et traitement des données
        energy_kcal = parse_input(str(data['energy-kcal']), 'energy-kcal_100g')
        saturated_fat = parse_input(str(data['saturated-fat']), 'saturated-fat_100g')
        sugars = parse_input(str(data['sugars']), 'sugars_100g')
        fiber = parse_input(str(data['fiber']), 'fiber_100g')
        proteins = parse_input(str(data['proteins']), 'proteins_100g')
        salt = parse_input(str(data['salt']), 'salt_100g')
        fruits_vegetables_nuts = parse_input(str(data['fruits-vegetables-nuts-estimate-from-ingredients']), 
                                          'fruits-vegetables-nuts-estimate-from-ingredients_100g')
        
        # Préparation des données
        features = np.array([energy_kcal, saturated_fat, sugars, fiber, proteins, 
                           salt, fruits_vegetables_nuts] + [0] * 39).reshape(1, -1)
        
        # Normalisation et prédiction
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        # Calcul du grade
        if prediction <= -1:
            grade = 'A'
        elif 0 <= prediction <= 2:
            grade = 'B'
        elif 3 <= prediction <= 10:
            grade = 'C'
        elif 11 <= prediction <= 18:
            grade = 'D'
        else:
            grade = 'E'
            
        return jsonify({
            'nutriscore_grade': grade,
            'nutriscore_score': round(prediction)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)