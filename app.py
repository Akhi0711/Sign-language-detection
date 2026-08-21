import os

import numpy as np
import tensorflow as tf

from flask import (
    Flask,
    render_template,
    request,
    jsonify
)

from PIL import Image


# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)


# ==========================================
# SETTINGS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "sign_language_model.keras"
)

CLASS_FILE = os.path.join(
    BASE_DIR,
    "models",
    "class_names.txt"
)

IMG_SIZE = 96


# ==========================================
# CHECK MODEL
# ==========================================

if not os.path.exists(MODEL_PATH):

    print("ERROR: Model not found!")
    print(MODEL_PATH)
    exit()

if not os.path.exists(CLASS_FILE):

    print("ERROR: Class names file not found!")
    print(CLASS_FILE)
    exit()


# ==========================================
# LOAD MODEL
# ==========================================

print("========================================")
print("Loading Sign Language AI Model...")
print("========================================")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ==========================================
# LOAD CLASS NAMES
# ==========================================

with open(
    CLASS_FILE,
    "r"
) as file:

    class_names = [
        line.strip()
        for line in file
        if line.strip()
    ]


print("\nClasses:")

for index, name in enumerate(class_names):

    print(
        f"{index} = {name}"
    )


print(
    "\nNumber of classes:",
    len(class_names)
)


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================
# PREDICTION API
# ==========================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # --------------------------------------
    # Check image
    # --------------------------------------

    if "image" not in request.files:

        return jsonify({
            "error": "Please select an image."
        }), 400


    file = request.files["image"]


    if file.filename == "":

        return jsonify({
            "error": "Please select an image."
        }), 400


    try:

        # ==================================
        # LOAD IMAGE
        # ==================================

        image = Image.open(
            file.stream
        ).convert("RGB")


        # ==================================
        # RESIZE IMAGE
        # ==================================

        image = image.resize(
            (
                IMG_SIZE,
                IMG_SIZE
            )
        )


        # ==================================
        # CONVERT IMAGE TO ARRAY
        # ==================================

        image_array = np.array(
            image
        ).astype(
            "float32"
        )




        # ==================================
        # ADD BATCH DIMENSION
        # ==================================

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # ==================================
        # PREDICT
        # ==================================

        predictions = model.predict(
            image_array,
            verbose=0
        )


        # ==================================
        # GET PREDICTED CLASS
        # ==================================

        predicted_index = int(
            np.argmax(
                predictions[0]
            )
        )


        predicted_class = class_names[
            predicted_index
        ]


        # ==================================
        # GET CONFIDENCE
        # ==================================

        confidence = float(
            predictions[0][
                predicted_index
            ] * 100
        )


        # ==================================
        # PRINT RESULT IN TERMINAL
        # ==================================

        print(
            "\n========================================"
        )

        print(
            "SIGN LANGUAGE PREDICTION"
        )

        print(
            "========================================"
        )

        print(
            "Predicted Sign:",
            predicted_class
        )

        print(
            f"Confidence: {confidence:.2f}%"
        )

        print(
            "========================================\n"
        )


        # ==================================
        # SEND RESULT TO WEBSITE
        # ==================================

        return jsonify({

            "prediction":
                predicted_class,

            "confidence":
                f"{confidence:.2f}"

        })


    except Exception as e:

        print(
            "Prediction error:",
            str(e)
        )

        return jsonify({

            "error":
                str(e)

        }), 500


# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "SIGNVISION ASL DETECTION SYSTEM"
    )

    print(
        "========================================"
    )

    print(
        "Open in browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "========================================\n"
    )


    app.run(
        debug=True
    )