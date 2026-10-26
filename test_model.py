import joblib
from pathlib import Path

# LOAD MODEL

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "clickbait_model.joblib"

model = joblib.load(MODEL_PATH)


# HEADER

print("===================================")
print("       CLICKBAIT DETECTOR")
print("===================================")

# TEST HEADLINES

while True:

    headline = input(
        "\nEnter a headline (or type 'exit' to quit): "
    )

    if headline.lower() == "exit":

        print("Program closed.")

        break


    if not headline.strip():

        print("Please enter a headline.")

        continue
    # PREDICTION

    prediction = model.predict([headline])[0]

    probabilities = model.predict_proba([headline])[0]

    classes = list(model.classes_)

    confidence = (
        probabilities[classes.index(prediction)]
        * 100
    )
    # CONVERT 0/1 TO TE

    if int(prediction) == 1:

        result = "CLICKBAIT"

    else:

        result = "NON-CLICKBAIT"
    # DISPLAY RESULT

    print("\nPrediction:", result)

    print(
        f"Confidence: {confidence:.2f}%"
    )