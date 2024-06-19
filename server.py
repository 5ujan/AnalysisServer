from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import google.generativeai as genai

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
load_dotenv()

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
        df = pd.read_csv(filepath, delimiter=',')  # Set delimiter to '\t' for tab-separated values
        columns = df.columns.tolist()
        return jsonify({'filename': file.filename, 'columns': columns})

@app.route('/analyze', methods=['POST'])
def analyze_data():
    filename = request.json['filename']
    result = request.json['result']
    parameters = request.json['parameters']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath, delimiter=',')
    analysis_result = perform_regression(df, result, parameters)
    return jsonify(analysis_result)

@app.route('/correlate', methods=['POST'])
def correlate_data():
    filename = request.json['filename']
    result = request.json['result']
    parameters = request.json['parameters']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath, delimiter=',')
    correlation_result = perform_correlation(df, result, parameters)
    return jsonify(correlation_result)

@app.route('/trends', methods=['GET'])
def get_trends():
    filename = request.args.get('filename')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    model = genai.GenerativeModel('gemini-pro')

    # Example of trend extraction
    trends = []
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, max_output_tokens=2048)
    agent = create_csv_agent(llm, filepath, verbose=True, allow_dangerous_code=True)
    user_question = "what the trends in the data"
    response = model.generate_content(
            [user_question + "Based on the apriori association rules provided,find interesting trends,patterns and insights which can help the related organization in layman language. Use percentage metrices if you can. make it simple."]
        ).text
    trends.append(response)
    return jsonify({"trends": trends})

@app.route('/gemini',methods=['POST'])
def get_gemini():
    prompt = request.json['prompt']
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([prompt]).text
    return jsonify({"response": response})

def perform_regression(df, result, parameters):
    # Extract the independent variables (parameters) and the dependent variable (result)
    X = df[parameters].values
    y = df[result].values

    # Add a column of ones to X to account for the intercept term
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    # Perform the linear regression using the normal equation
    coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

    # Create a dictionary for the results
    result_dict = {param: round(coefficients[i + 1], 2) for i, param in enumerate(parameters)}
    return result_dict

def perform_correlation(df, result, parameters):
    correlations = {}
    for param in parameters:
        if param != result:
            correlation = df[param].corr(df[result])
            correlations[param] = f"{correlation:.2f}"
    return correlations

if __name__ == '__main__':
    app.run(debug=True)
