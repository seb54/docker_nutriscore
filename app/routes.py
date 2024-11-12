from flask import Blueprint, render_template, request, jsonify
from app.graphs import create_graph
import requests
import os

main = Blueprint('main', __name__)
MODEL_SERVICE_URL = os.environ.get('MODEL_SERVICE_URL', 'http://modelservice:5001')

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

@main.route('/')
def dashboard():
    graph = create_graph()
    return render_template('index.html', graph=graph)

@main.route('/predict')
def predict():
    return render_template('predict.html')

@main.route('/results', methods=['POST'])
def results():
    try:
        # Préparation des données pour l'API
        data = {
            'energy-kcal': request.form.get('energy-kcal'),
            'saturated-fat': request.form.get('saturated-fat'),
            'sugars': request.form.get('sugars'),
            'fiber': request.form.get('fiber'),
            'proteins': request.form.get('proteins'),
            'salt': request.form.get('salt'),
            'fruits-vegetables-nuts-estimate-from-ingredients': request.form.get('fruits-vegetables-nuts'),
            'selected_name': request.form.get('selected_name', '')
        }

        # Appel au service de modèle
        response = requests.post(f"{MODEL_SERVICE_URL}/predict", json=data)
        result = response.json()
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 400
            
        return render_template('results.html', 
                             y_new_pred=result['nutriscore_grade'], 
                             score_new_pred=result['nutriscore_score'])
                             
    except Exception as e:
        return jsonify({'error': str(e)}), 500
