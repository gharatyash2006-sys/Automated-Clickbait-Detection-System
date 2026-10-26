import joblib
from pathlib import Path
from flask import Flask, render_template, request

app = Flask(__name__)


# LOAD MODEL

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "clickbait_svm_model.joblib"
)

model = joblib.load(MODEL_PATH)



# HOME PAGE


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    headline = ""

    if request.method == "POST":

        headline = request.form.get(
            "headline",
            ""
        ).strip()

        if headline:

            # Get prediction
            raw_prediction = model.predict(
                [headline]
            )[0]

            # Get probabilities
            probabilities = model.predict_proba(
                [headline]
            )[0]

            # Get model classes
            classes = list(
                model.classes_
            )

            # Confidence of predicted class
            confidence = (
                probabilities[
                    classes.index(raw_prediction)
                ] * 100
            )

            # Convert 0/1 into readable labels
            if int(raw_prediction) == 1:

                prediction = "clickbait"

            else:

                prediction = "non-clickbait"


    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        headline=headline
    )



# RUN APPLICATION


if __name__ == "__main__":

    app.run(
        debug=True
    )