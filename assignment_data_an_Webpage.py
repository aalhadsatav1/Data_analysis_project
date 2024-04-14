import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
from flask import Flask, request, render_template_string

data_path = 'C:\\Users\\aalha\\Desktop\\healthcare_dataset.csv' 
data = pd.read_csv(data_path)
required_columns = ['Blood Group Type', 'Gender', 'Medical Condition']
data = data[required_columns]

encoders = {col: LabelEncoder() for col in required_columns}
for col in required_columns:
    data[col] = encoders[col].fit_transform(data[col])

X = data[['Blood Group Type', 'Gender']]  # Features
y = data['Medical Condition']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
for col, encoder in encoders.items():
    with open(f'{col}_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)

submissions_path = 'C:\\Users\\aalha\\Desktop\\submissions.csv'  # We can change the path

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
encoders = {col: pickle.load(open(f'{col}_encoder.pkl', 'rb')) for col in ['Blood Group Type', 'Gender', 'Medical Condition']}

HTML_FORM = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Risk Prediction Form</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        form { background-color: #f2f2f2; padding: 20px; border-radius: 5px; }
        input[type=text], select { width: 100%; padding: 12px 20px; margin: 8px 0; display: inline-block; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
        input[type=submit] { width: 100%; background-color: #4CAF50; color: white; padding: 14px 20px; margin: 8px 0; border: none; border-radius: 4px; cursor: pointer; }
        input[type=submit]:hover { background-color: #45a049; }
    </style>
</head>
<body>
<h2>Enter Your Health Details</h2>
<form action="/" method="post">
    <label for="name">Name:</label><br>
    <input type="text" id="name" name="name" placeholder="Your name..."><br>
    <label for="gender">Gender:</label><br>
    <select id="gender" name="gender">
        <option value="Male">Male</option>
        <option value="Female">Female</option>
        <option value="Other">Other</option>
    </select><br>
    <label for="blood_group">Blood Group:</label><br>
    <select id="blood_group" name="blood_group">
        <option value="A+">A+</option>
        <option value="A-">A-</option>
        <option value="B+">B+</option>
        <option value="B-">B-</option>
        <option value="AB+">AB+</option>
        <option value="AB-">AB-</option>
        <option value="O+">O+</option>
        <option value="O-">O-</option>
    </select><br><br>
    <input type="submit" value="Submit">
</form>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        name = request.form.get('name')
        gender = request.form.get('gender')
        blood_group = request.form.get('blood_group')

        input_features = pd.DataFrame([[blood_group, gender]], columns=['Blood Group Type', 'Gender'])
        input_features['Blood Group Type'] = encoders['Blood Group Type'].transform([blood_group])
        input_features['Gender'] = encoders['Gender'].transform([gender])

        probabilities = model.predict_proba(input_features)[0]
        top_indices = probabilities.argsort()[-3:][::-1] 
        conditions = [(encoders['Medical Condition'].inverse_transform([idx])[0], prob * 100) for idx, prob in zip(top_indices, probabilities[top_indices])]
        predictions = ", ".join([f"{cond[0]} (Risk: {cond[1]:.2f}%)" for cond in conditions])
        predicted_condition = conditions[0][0]
        
        df = pd.DataFrame([[name, gender, blood_group, predicted_condition]], columns=['Name', 'Gender', 'Blood Group Type', 'Predicted Medical Condition'])
        df.to_csv(submissions_path, mode='a', header=not os.path.exists(submissions_path), index=False)

        return f'<p>Thank you, {name}. Based on your input, the medical conditions you are at most risk of are: {predictions}.</p>'
    else:
        return render_template_string(HTML_FORM)

if __name__ == '__main__':
    app.run(debug=True)
