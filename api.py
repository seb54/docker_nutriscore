from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Chargement des données pour la médiane
df = pd.read_csv("cleaned_data.csv", sep=',', on_bad_lines='skip', low_memory=False)

# Chargement du modèle et du scaler
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

def parse_input(value, nom):
    global df
    try:
        if value is None or value == "" or value == "NaN":
            return df[nom].median()
        return float(value)
    except (ValueError, KeyError):
        return 0.0

def score_to_grade(score):
    if score <= -1:
        return 'A'
    elif 0 <= score <= 2:
        return 'B'
    elif 3 <= score <= 10:
        return 'C'
    elif 11 <= score <= 18:
        return 'D'
    else:
        return 'E'

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validation des données
        required_fields = ['energy-kcal', 'saturated-fat', 'sugars', 
                         'fiber', 'proteins', 'salt', 
                         'fruits-vegetables-nuts-estimate-from-ingredients',
                         'selected_name']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Le champ {field} est manquant'}), 400
        
        # Traitement des données avec parse_input
        energy_kcal = parse_input(str(data['energy-kcal']), 'energy-kcal_100g')
        saturated_fat = parse_input(str(data['saturated-fat']), 'saturated-fat_100g')
        sugars = parse_input(str(data['sugars']), 'sugars_100g')
        fiber = parse_input(str(data['fiber']), 'fiber_100g')
        proteins = parse_input(str(data['proteins']), 'proteins_100g')
        salt = parse_input(str(data['salt']), 'salt_100g')
        fruits_vegetables_nuts_estimate = parse_input(
            str(data['fruits-vegetables-nuts-estimate-from-ingredients']),
            'fruits-vegetables-nuts-estimate-from-ingredients_100g'
        )
            
        # Création du tableau de données
        features = [energy_kcal, saturated_fat, sugars, fiber, 
                   proteins, salt, fruits_vegetables_nuts_estimate] + [0] * 39
        
        # Prédiction
        new_data_scaled = scaler.transform([features])
        prediction = model.predict(new_data_scaled)
        score = round(prediction[0])
        grade = score_to_grade(score)
        
        return jsonify({
            'nutriscore_grade': grade,
            'nutriscore_score': score
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)