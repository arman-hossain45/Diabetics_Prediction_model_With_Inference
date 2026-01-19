import gradio as gr
import pandas as pd
import pickle
import numpy as np

# ==============================
# 1. Load the trained model
# ==============================
with open("diabetics.pkl", "rb") as f:
    model = pickle.load(f)

# ==============================
# 2. Prediction Logic
# ==============================
def predict_dia(
    Pregnancies,
    Glucose,
    BloodPressure,
    SkinThickness,
    Insulin,
    BMI,
    DiabetesPedigreeFunction,
    Age
):

    # -------- Safety Check --------
    inputs = [
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age
    ]

    
    # -------- Create DataFrame (same column order as training) --------
    input_df = pd.DataFrame([[
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DiabetesPedigreeFunction,
        Age
    ]], columns=[
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age'
    ])

    # -------- Prediction --------
    prediction = model.predict(input_df)[0]

    # -------- Output --------
    if prediction == 1:
        return "Result: Diabetic (Yes)"
    else:
        return "Result: Not Diabetic (No)"

# ==============================
# 3. Gradio Interface
# ==============================
inputs = [
    gr.Number(label="Pregnancies"),
    gr.Number(label="Glucose"),
    gr.Number(label="BloodPressure"),
    gr.Number(label="SkinThickness"),
    gr.Number(label="Insulin"),
    gr.Number(label="BMI"),
    gr.Number(label="DiabetesPedigreeFunction"),
    gr.Number(label="Age")
]

app = gr.Interface(
    fn=predict_dia,
    inputs=inputs,
    outputs="text",
    title="Diabetics Predictor"
)

# ==============================
# 4. Launch App
# ==============================
app.launch(share=True)
