from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        df = pd.read_csv(filepath, delimiter='\t')  # Set delimiter to '\t' for tab-separated values
        columns = df.columns.tolist()
        return jsonify({'filename': file.filename, 'columns': columns})

@app.route('/analyze', methods=['POST'])
def analyze_data():
    filename = request.json['filename']
    result = request.json['result']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath, delimiter='\t')
    
    analysis_result = perform_regression(filepath, result)
    return jsonify(analysis_result)

def perform_regression(filepath, result):
    # Load the CSV file
    df = pd.read_csv(filepath, delimiter='\t')

    # Find the index of the "result" column
    result_index = df.columns.tolist().index(result)

    # Extract the independent variables (parameters) and the dependent variable (result)
    param_columns = [col for i, col in enumerate(df.columns) if i != result_index]
    X = df[param_columns].values
    y = df.iloc[:, result_index].values

    # Add a column of ones to X to account for the intercept term
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    # Perform the linear regression using the normal equation
    coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

    # Create a dictionary for the results
    result_dict = {param: int(coefficients[i + 1] * 10) for i, param in enumerate(param_columns)}
    return result_dict

if __name__ == '__main__':
    app.run(debug=True)
