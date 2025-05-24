# Car-price-prediction-
Built a web application that predicts used car prices based on key features like age, kilometers driven, fuel type, and transmission. Implemented ensemble learning with a Voting Regressor combining Random Forest and Gradient Boosting, and integrated it with a Flask-based interface for real-time prediction.

# 🚗 Car Price Prediction Web App

This project is a machine learning-based web application that predicts the resale price of a used car based on various user inputs. The application is built using Flask and deployed via Ngrok for live usage.

---

## 🔍 Overview

The primary goal of this project is to help car owners estimate a fair resale value of their used car by analyzing features such as:

- Year of manufacture
- price of the car
- Kilometers driven
- Number of previous owners
- Fuel type (Petrol/Diesel)
- Seller type (Dealer/Individual)
- Transmission type (Manual/Automatic)

The machine learning model is trained on a real-world dataset using ensemble regression methods to provide accurate price predictions.

---

## 🚀 Features

- **Interactive Web Interface** built with HTML and Flask
- **Data Preprocessing** with Pandas (feature engineering, one-hot encoding)
- **Machine Learning Model**: Voting Regressor (Random Forest + Gradient Boosting)
- **Real-time Predictions**
- **Ngrok Tunnel** used to expose local Flask app for live demo

---

## 🛠️ Tech Stack

| Category        | Technologies Used                                  |
|----------------|------------------------------------------------------|
| Backend         | Python, Flask                                       |
| Frontend        | HTML, CSS                                           |
| Machine Learning| scikit-learn, Pandas, NumPy                         |
| Deployment      | Ngrok (for public URL tunneling)                    |
| Models Used     | RandomForestRegressor, GradientBoostingRegressor    |


---


