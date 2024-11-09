import random
import pandas as pd
import os, cohere, markdown, joblib
from flask_sqlalchemy import SQLAlchemy
from sklearn.preprocessing import LabelEncoder
from flask import Flask, jsonify, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.sqlite3'

db=SQLAlchemy(app)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fallback_secret_key')
co = cohere.Client('9xtb0BFwgzrTgGQIG7QJrdZAUBH8oQoW30EK2Z7I')

model = joblib.load('./models/gut_health_model.pkl')
scaler = joblib.load('./models/scaler.pkl')
feature_columns = joblib.load('./models/feature_columns.pkl')
encoder = LabelEncoder()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    date_joined = db.Column(db.DateTime, default=db.func.current_timestamp())

app.app_context().push()
db.create_all()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        new_user=User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()        
        return redirect(url_for('signin'))
    
    return render_template('./templates/sign-up.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Authenticates user
        existing_user=User.query.filter_by(username=username, password=password).first()
        if existing_user:
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('./templates/not-found.html')
    
    return render_template('./templates/sign-in.html')

@app.route('/signout', methods=['GET', 'POST'])
def signout():
    # Clears the session
    session.clear()
    return redirect(url_for('signin'))

@app.route('/about', methods=['GET'])
def about():
    if request.method == 'GET':
        return render_template('./templates/about.html')
    
@app.route('/contact', methods=['GET'])
def contact():
    if request.method == 'GET':
        return render_template('./templates/contact.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('signin'))
    
    username = session['username']
    return render_template('./templates/dashboard.html', username=username)

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('signin'))
    
    data = {
        'Gender': [request.form['Gender']],
        'Age': [float(request.form['Age'])],
        'Height': [float(request.form['Height'])],
        'Weight': [float(request.form['Weight'])],
        'family_history_with_overweight': [request.form['family_history_with_overweight']],
        'FAVC': [request.form['FAVC']],
        'FCVC': [float(request.form['FCVC'])],
        'NCP': [float(request.form['NCP'])],
        'CAEC': [request.form['CAEC']],
        'SMOKE': [request.form['SMOKE']],
        'CH2O': [float(request.form['CH2O'])],
        'SCC': [request.form['SCC']],
        'FAF': [int(request.form['FAF'])],
        'TUE': [int(request.form['TUE'])],
        'CALC': [request.form['CALC']],
        'MTRANS': [request.form['MTRANS']],
        'NObeyesdad': [request.form['NObeyesdad']]
    }

    
    # Creates a DataFrame from input
    new_data = pd.DataFrame(data)

    label_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    for col in label_cols:
        new_data[col] = encoder.fit_transform(new_data[col])

    new_data = pd.get_dummies(new_data, columns=['MTRANS'], drop_first=False)
    new_data = pd.get_dummies(new_data, columns=['CAEC'], drop_first=False)
    new_data = pd.get_dummies(new_data, columns=['CALC'], drop_first=False)
    new_data = pd.get_dummies(new_data, columns=['NObeyesdad'], drop_first=False)

    missing_cols = set(feature_columns) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0

    # Ensures column order matches the training data
    new_data = new_data[feature_columns]

    gut_health_columns = ['NObeyesdad_Insufficient_Weight', 'NObeyesdad_Normal_Weight', 'NObeyesdad_Obesity_Type_I', 'NObeyesdad_Obesity_Type_II', 'NObeyesdad_Obesity_Type_III', 'NObeyesdad_Overweight_Level_I', 'NObeyesdad_Overweight_Level_II', 'MTRANS_Automobile', 'MTRANS_Public_Transportation', 'MTRANS_Bike', 'MTRANS_Motorbike', 'MTRANS_Walking', 'CALC_Frequently', 'CALC_no', 'CALC_Sometimes', 'CAEC_Always', 'CAEC_Frequently', 'CAEC_no', 'CAEC_Sometimes']
    new_data[gut_health_columns] = new_data[gut_health_columns].astype(int)

    # Scales the features
    new_data_scaled = scaler.transform(new_data)

    y_pred_new = model.predict(new_data_scaled)

    session['new_data'] = new_data.to_json()

    # Outputs the predicted gut health
    if y_pred_new[0][0] == 1:
        result = "Good"
    elif y_pred_new[0][1] == 1:
        result = "Moderate"
    else:
        result = "Poor"

    # Generates a personalized health report using Generative AI (Cohere)
    prompt = f"""
    Generate a personalized health report based on the following user data:
    Gender: {request.form['Gender']}
    Age: {request.form['Age']}
    Height: {request.form['Height']} meters
    Weight: {request.form['Weight']} kg
    Family history of overweight: {request.form['family_history_with_overweight']}
    Physical Activity Frequency: {request.form['FAF']} times per week
    Diet-related habits (FAVC, FCVC, etc.): FAVC - {request.form['FAVC']}, FCVC - {request.form['FCVC']}
    Smoking habit: {request.form['SMOKE']}
    Daily water intake (CH2O): {request.form['CH2O']} liters
    TUE (Technology usage time): {request.form['TUE']} hours

    Provide a detailed health report with the following:
    1. Evaluation of current health based on the user's data.
    2. Personalized diet suggestions and exercise recommendations based on the age, weight, height, and activity levels.
    3. Long-term health improvement tips based on the data (including managing weight, increasing physical activity, dietary improvements, and stress reduction).
    
    Restructure it with Bullet Points
    """

    response = co.generate(
        model="command-r-plus-08-2024",
        prompt=prompt,
        max_tokens=1000
    )

    health_report = response.generations[0].text.strip()
    health_report = markdown.markdown(health_report)
    username = session['username']

    return render_template('./templates/dashboard.html', prediction=result, health_report=health_report, username=username)

@app.route('/inference', methods=['GET', 'POST'])
def inference():
    new_data_json = session.get('new_data')
    new_data = pd.read_json(new_data_json)

    trend_data = []  
    new_data_copy = new_data.copy()  

    for month in range(1, 13):  
        new_data_copy['Age'] += 1/12  
        new_data_copy['Weight'] += random.uniform(-0.1, 0.1)  # Minor Weight Fluctuations
        new_data_copy['FCVC'] = new_data_copy['FCVC'].apply(lambda x: max(1, min(3, x + random.uniform(-0.1, 0.1))))
        new_data_copy['FAF'] = new_data_copy['FAF'].apply(lambda x: max(0, x + random.choice([-1, 0, 1])))  # Activity Fluctuations
        
        new_data_scaled_copy = scaler.transform(new_data_copy) 

        y_pred_new_copy = model.predict(new_data_scaled_copy)

        if y_pred_new_copy[0][0] == 1:
            risk_score = "Good"
        elif y_pred_new_copy[0][1] == 1:
            risk_score = "Moderate"
        else:
            risk_score = "Poor"
        
        trend_data.append({"month": month, "risk_score": risk_score})

    return jsonify({"trend_data": trend_data})

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    return render_template("./templates/forecast.html")

if __name__ == '__main__':
    from os import environ
    port = int(environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)