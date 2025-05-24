from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load and preprocess dataset
data = pd.read_csv(r"D:\Sem 6\Car-Price-Prediction-main\car data.csv")  # Use 'r' for raw string to avoid path issues

# Feature Engineering
data['Age'] = 2024 - data['Year']
data = data.drop(['Year', 'Car_Name'], axis=1)
data = pd.get_dummies(data, drop_first=True)

# Splitting the data
X = data.drop("Selling_Price", axis=1)
y = data["Selling_Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model = VotingRegressor([('rf', rf), ('gb', gb)])

model.fit(X_train, y_train)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            print(request.form)  # Debugging: Print form data
            
            # Collect form data safely
            year = int(request.form.get('Year', 2024))
            present_price = float(request.form.get('Present_Price', 0))
            kms_driven = int(request.form.get('Kms_Driven', 0))
            owner = int(request.form.get('Owner', 0))
            
            # Fuel type encoding
            fuel_type = request.form.get('Fuel_Type', 'Petrol')
            fuel_petrol = 1 if fuel_type == 'Petrol' else 0
            fuel_diesel = 1 if fuel_type == 'Diesel' else 0
            
            # Seller type encoding
            seller_type = 1 if request.form.get('Seller_Type', 'Dealer') == 'Individual' else 0
            
            # Transmission encoding
            transmission = 1 if request.form.get('Transmission', 'Manual') == 'Manual' else 0
            
            # Feature vector
            features = np.array([[2024 - year, present_price, kms_driven, owner, fuel_petrol, fuel_diesel, seller_type, transmission]])
            features = scaler.transform(features)  # Apply scaling
            
            # Predict selling price
            prediction = model.predict(features)
            output = round(prediction[0], 2)
            
            return render_template('index.html', prediction_text=f"You can sell the Car at {output} lakhs")

    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', prediction_text="Error in prediction. Check input values.")

if __name__ == "__main__":
    app.run(debug=True)
