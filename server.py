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
from trends import getTrends
from sklearn.preprocessing import MinMaxScaler
import json

import warnings
from MLP import data_maker, MLP, modelselector
warnings.filterwarnings("ignore", category=DeprecationWarning)
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

load_dotenv()
model = None
filepath = ''
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    # Cleaning unnecessary columns
    
    id_columns = [col for col in data.columns if 'id' in col.lower()]
    name_columns = [col for col in data.columns if 'name' in col.lower()]
    total_columns = [col for col in data.columns if 'total' in col.lower()]
    result_columns = [col for col in data.columns if 'outcome'  in col.lower()]
    columns_to_drop =  id_columns + name_columns + total_columns+result_columns
    data.drop(columns_to_drop, axis=1, inplace=True)
    # Convert object columns to categorical columns
    object_columns = data.select_dtypes(include=['object']).columns
    for col in object_columns:
        data[col] = data[col].astype('category')
    # Handle missing values for numeric columns (impute with mean)
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    # Handle missing values for categorical columns (impute with mode)
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].fillna(data[col].mode()[0])
    return data


@app.route('/upload', methods=['POST'])
def upload_file():
    global filepath
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        df = pd.read_csv(filepath, delimiter=',')  # Set delimiter to '\t' for tab-separated values
        columns = df.select_dtypes(include=['number']).columns.tolist()
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

@app.route('/trends', methods=['POST'])
def get_trends():
    print(request.get_json())
    print("reaches here")
    filename = request.json['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    model = genai.GenerativeModel('gemini-pro')

    # Example of trend extraction
    
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, max_output_tokens=2048)
    # agent = create_csv_agent(llm, filepath, verbose=True, allow_dangerous_code=True)
    user_question = "what the trends in the data"
    print(12121211341)


    data = load_data(filepath)
    data = clean_data(data)


    aprioriTrend = getTrends(data) 
    
    at= json.dumps(aprioriTrend)
    print(aprioriTrend)
    response = aprioriTrend
    print("reaches here")
    response = model.generate_content(
            [user_question + "Based on the apriori association rules provided,find interesting trends,patterns and insights which can help the related organization in layman language. Use percentage metrices if you can. make it simple: \n Apriori association:"+ at ]
        ).text
    
    return jsonify({"trends": response})

@app.route('/gemini',methods=['POST'])
def get_gemini():
    prompt = request.json['prompt']
    prev = request.json['previous']
    prev = prev[-2:] if len(prev) >= 2 else prev
    prev = json.dumps(prev)
    msg = "Last two responses were:"+ prev+ "New prompt:"+prompt
    # print(msg)
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([msg]).text
    return jsonify({"response": response})

@app.route('/trendsquery',methods=['POST'])
def get_trendquery():
    global filepath
    prompt = request.json['prompt']
    prev = request.json['previous']
    prev = prev[-2:] if len(prev) >= 2 else prev
    prev = json.dumps(prev)
    # print(msg)
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-pro')
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, max_output_tokens=2048)
    agent = create_csv_agent(llm, filepath, verbose=True, allow_dangerous_code=True)
    chain = agent.run(prompt)
    msg = "Last two responses were:"+ prev+ "New prompt:"+prompt+"langchain output:"+chain
    response = model.generate_content([msg]).text
    return jsonify({"response": response})


@app.route('/train', methods=['POST'])
def get_model():
    global model
    filename = request.json['filename']
    selectedparam = request.json['selectedParams']
    selectedcolumn = request.json['selectedColumn']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    column_indices = []
    df = pd.read_csv(filepath, delimiter=',')
    for param in selectedparam:
            index = df.columns.get_loc(param)
            column_indices.append(index)
    resultindex =df.columns.get_loc(selectedcolumn)
    result =[resultindex]

    xnum, xcat, y, scalerx, scalery, encoding_dicts = data_maker(column_indices, result, filepath)
    model=MLP(xnum, xcat, y, scalerx, scalery, encoding_dicts,2)
    model.train_model()
    Val_loss = model.validation_loss_val
    numpy_value = Val_loss.numpy()
    scalar_value = numpy_value.item()
    print(scalar_value)
    return jsonify({"response": scalar_value})
@app.route('/predict',methods =['POST'])
def give_prediction():
    fieldValues = request.json['fieldValues']
    print(fieldValues)
    values_list = [float(value) for value in fieldValues.values()]
    print(values_list)
    predicted = model.generate(values_list)
    return jsonify({"response": predicted[0][0]})

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
