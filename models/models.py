import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

filename = './data/train.csv'
train = pd.read_csv(filename)
train_data = train.drop('id', axis=1)

def calculate_gut_health(row):
    score = 0

    # Gender
    score += 1 if row['Gender'] == 'Female' else 0

   # Age
    if 18 <= row['Age'] <= 30:
        score += 2
    elif 31 <= row['Age'] <= 50:
        score += 1
    elif 51 <= row['Age'] <= 61:
        score += 0
    else:
        score -= 1

   # BMI (using Height and Weight to calculate BMI)
    bmi = row['Weight'] / (row['Height'] ** 2)

    # Scoring based on BMI range
    if 18.5 <= bmi <= 24.9:
        score += 2  # Normal weight
    elif 25 <= bmi <= 29.9:
        score += 1  # Overweight
    elif 30 <= bmi <= 34.9:
        score = 0  # Obesity class 1 (mild obesity)
    elif 35 <= bmi <= 39.9:
        score = -1  # Obesity class 2 (moderate obesity)
    elif 40 <= bmi <= 79:
        score = -2  # Obesity class 3 (severe obesity or extreme obesity)
    else:
        score = -3  # Extremely low BMI

    # Family History with Overweight
    score += 1 if row['family_history_with_overweight'] == 'no' else -1

    # FAVC (Frequent High-Caloric Food Consumption)
    score += 2 if row['FAVC'] == 'no' else -1

    # FCVC (Frequency of Vegetable Consumption)
    if row['FCVC'] >= 2.5:
        score += 2
    elif 1.5 <= row['FCVC'] < 2.5:
        score += 1
    else:
        score -= 1

    # NCP (Number of Meals per Day)
    if 3 <= row['NCP'] <= 4:
        score += 2
    elif 2 <= row['NCP'] < 3:
        score += 1
    else:
        score -= 1

    # CAEC (Consumption Between Meals)
    if row['CAEC'] == 'no':
        score += 2
    elif row['CAEC'] == 'Sometimes':
        score += 1
    elif row['CAEC'] == 'Frequently':
        score -= 1
    else:
        score -= 2

    # SMOKE
    score += 2 if row['SMOKE'] == 'no' else -2

    # CH2O (Daily Water Intake)
    score += 2 if row['CH2O'] >= 2.5 else (1 if row['CH2O'] >= 2.0 else -1)

    # SCC (Self-Monitoring Calories)
    score += 1 if row['SCC'] == 'yes' else 0

    # FAF (Physical Activity Frequency)
    score += 2 if row['FAF'] >= 2.5 else (1 if row['FAF'] >= 1.0 else -1)

    # TUE (Technology Usage)
    score += 2 if row['TUE'] <= 1 else (1 if row['TUE'] <= 1.5 else -1)

    # CALC (Alcohol Consumption)
    score += 2 if row['CALC'] == 'no' else (-1 if row['CALC'] == 'Frequently' else 1)

    # MTRANS (Transportation Mode)
    score += 2 if row['MTRANS'] == 'Walking' else (1 if row['MTRANS'] == 'Bike' or row['MTRANS'] == 'Public_Transportation' else 0)

    # NObeyesdad (Obesity Classification)
    score += 2 if row['NObeyesdad'] == 'Normal_Weight' else (-1 if row['NObeyesdad'].startswith('Obesity') else 1)

    if score >= 15:
        return 'Good'
    elif score >= 0:
        return 'Moderate'
    else:
        return 'Poor'

# Applys the function to create the 'Gut Health' column
train_data['Gut_Health'] = train_data.apply(calculate_gut_health, axis=1)
data = train_data

# Handles categorical variables
label_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']  # Add all categorical columns here
encoder = LabelEncoder()
for col in label_cols:
    data[col] = encoder.fit_transform(data[col])

# One-hot encoding for multi-class columns (e.g., 'MTRANS' with multiple categories)
data = pd.get_dummies(data, columns=['MTRANS'], drop_first=False)
data = pd.get_dummies(data, columns=['CAEC'], drop_first=False)
data = pd.get_dummies(data, columns=['CALC'], drop_first=False)
data = pd.get_dummies(data, columns=['NObeyesdad'], drop_first=False)
data = pd.get_dummies(data, columns=['Gut_Health'], drop_first=False)

gut_health_columns = ['Gut_Health_Good', 'Gut_Health_Moderate', 'Gut_Health_Poor', 'NObeyesdad_Insufficient_Weight', 'NObeyesdad_Normal_Weight', 'NObeyesdad_Obesity_Type_I', 'NObeyesdad_Obesity_Type_II', 'NObeyesdad_Obesity_Type_III', 'NObeyesdad_Overweight_Level_I', 'NObeyesdad_Overweight_Level_II', 'MTRANS_Automobile', 'MTRANS_Public_Transportation', 'MTRANS_Bike', 'MTRANS_Motorbike', 'MTRANS_Walking', 'CALC_Frequently', 'CALC_no', 'CALC_Sometimes', 'CAEC_Always', 'CAEC_Frequently', 'CAEC_no', 'CAEC_Sometimes']
data[gut_health_columns] = data[gut_health_columns].astype(int)

# Separates features (X) and target (y)
X = data.drop(columns=['Gut_Health_Good', 'Gut_Health_Moderate', 'Gut_Health_Poor'])
y = data[['Gut_Health_Good', 'Gut_Health_Moderate', 'Gut_Health_Poor']]

# Standardizes the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splits the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Trains a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Makes predictions
y_pred = model.predict(X_test)

y_test_np = y_test.to_numpy()
y_pred_np = y_pred

# Evaluates the model
accuracy = accuracy_score(y_test_np, y_pred_np)
conf_matrix = confusion_matrix(y_test_np.argmax(axis=1), y_pred_np.argmax(axis=1))
class_report = classification_report(y_test_np.argmax(axis=1), y_pred_np.argmax(axis=1))

# Outputs results
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

# Saves the model
joblib.dump(model, 'gut_health_model.pkl', compress=3)

# Saves the scaler
joblib.dump(scaler, 'scaler.pkl')

feature_columns = X.columns  # Column names from training data
joblib.dump(feature_columns, 'feature_columns.pkl')