
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load Model
model = joblib.load("../model/best_model.pkl")
mlb = joblib.load("../model/multilabel_encoder.pkl")
label_encoder = joblib.load("../model/disease_encoder.pkl")

# Load Supporting Data
description_df = pd.read_csv("../dataset/symptom_Description.csv")
precaution_df = pd.read_csv("../dataset/symptom_precaution.csv")

description_df["Disease"] = description_df["Disease"].str.strip()
precaution_df["Disease"] = precaution_df["Disease"].str.strip()

description_dict = dict(zip(description_df["Disease"], description_df["Description"]))

precaution_df = precaution_df.fillna("Not Available")

precaution_dict = {}

for _, row in precaution_df.iterrows():
    precaution_dict[row["Disease"]] = [
        row["Precaution_1"],
        row["Precaution_2"],
        row["Precaution_3"],
        row["Precaution_4"]
    ]

all_symptoms = sorted(list(mlb.classes_))


@app.route("/")
def home():
    return render_template(
        "index.html",
        symptoms=all_symptoms
    )


@app.route("/predict", methods=["POST"])
def predict():

    symptoms = request.form.getlist("symptoms")

    input_vector = mlb.transform([symptoms])

    prediction = model.predict(input_vector)

    disease = label_encoder.inverse_transform(prediction)[0].strip()

    description = description_dict.get(disease, "Description not available.")

    precautions = precaution_dict.get(disease, [])

    return render_template(
        "result.html",
        disease=disease,
        description=description,
        precautions=precautions
    )


if __name__ == "__main__":
    app.run(debug=True)
