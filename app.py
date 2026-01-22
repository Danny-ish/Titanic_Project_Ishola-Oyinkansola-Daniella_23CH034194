from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model & preprocessors
model_path = os.path.join('model', 'titanic_survival_model.pkl')
scaler_path = os.path.join('model', 'scaler.pkl')
le_sex_path = os.path.join('model', 'le_sex.pkl')
le_emb_path = os.path.join('model', 'le_embarked.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
with open(le_sex_path, 'rb') as f:
    le_sex = pickle.load(f)
with open(le_emb_path, 'rb') as f:
    le_emb = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            pclass = int(request.form['pclass'])
            sex = request.form['sex']
            age = float(request.form['age'])
            fare = float(request.form['fare'])
            embarked = request.form['embarked']

            # Encode
            sex_enc = le_sex.transform([sex])[0]
            emb_enc = le_emb.transform([embarked])[0]

            # Create array in correct order
            input_data = np.array([[pclass, sex_enc, age, fare, emb_enc]])

            # Scale
            input_scaled = scaler.transform(input_data)

            # Predict
            pred = model.predict(input_scaled)[0]
            prediction = "Survived" if pred == 1 else "Did Not Survive"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)   # ← important change