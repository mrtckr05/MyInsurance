import pickle
import pandas as pd
import os
from model import predicted_prices
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget
from insurance_gui import Ui_MyInsurance
import numpy as np

def load_model():
    """
    Load the trained model, scaler, and feature names from pickle files
    """
    model_path = "D:\\vscode-works\\insurance_prediction_app\\models\\insurance_model.pkl"
    scaler_path = "D:\\vscode-works\\insurance_prediction_app\\models\\feature_scaler.pkl"
    encoder_path = "D:\\vscode-works\\insurance_prediction_app\\models\\feature_encoder.pkl"
    features_path = "D:\\vscode-works\\insurance_prediction_app\\models\\feature_names.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(features_path):
        print("Model files not found. Please run save_model.py first.")
        sys.exit(1)

    with open(model_path, 'rb') as f:
        modele = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)  
    
    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)

    return modele, scaler, encoder, feature_names
       

def main():
    """
    Main function to run the insurance prediction application
    """
    print("Starting the insurance prediction application...")
    app = QtWidgets.QApplication(sys.argv)
    MyInsurance = QtWidgets.QMainWindow()
    ui = Ui_MyInsurance(age=None, bmi=None, sex=None, children=None, smoker=None, region=None, price=None)
    ui.setupUi(MyInsurance)
    MyInsurance.show()
        
    modele, scaler, encoder, feature_names = load_model()
    
    def display_price():        
        #Function to handle button click event for prediction
        print("Button clicked, starting prediction thread...")
                
        def get_inputs():
            #Function to get user inputs from the UI
            print("Getting user inputs...")                
            age = ui.get_age()           
            bmi = ui.get_bmi()
            sex = ui.sex
            children = ui.get_children()           
            smoker = ui.get_smoker()   
            region = ui.region
            
            # Validate inputs             
            input_data = [age, bmi, sex, children, smoker, region[0], region[1], region[2]]
            user_input = np.array(input_data)
            print("User inputs collected:", user_input)
            return user_input

        user_input = get_inputs()
        price = predicted_prices(modele, scaler, user_input)
        print("price calculated!!!", price)
        ui.display_price.setText(f"Predicted Insurance Price: ${price:.2f}")
        print("Prediction displayed on UI.")
    def is_sender_clicked():
        print("Sender clicked")
        display_price()

    ui.sender.clicked.connect(is_sender_clicked)

    MyInsurance.show()
    sys.exit(app.exec_()) 

    
if __name__ == "__main__":
    main()   