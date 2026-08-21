# SignVision - ASL Sign Language Detection

## 📌 Project Overview

**SignVision** is a Machine Learning-based American Sign Language (ASL) detection system that recognizes hand signs from images.

The system uses a Convolutional Neural Network (CNN) trained on the ASL Alphabet dataset and provides a web-based interface built using Flask.

Users can upload an image of a hand sign, and the trained AI model predicts the corresponding ASL class along with its confidence score.

---

## 🎯 Objectives

The main objectives of this project are:

- To develop an AI-based sign language recognition system.
- To recognize ASL alphabet hand gestures from images.
- To classify images into 29 different ASL classes.
- To provide a simple web interface for image-based prediction.
- To display the predicted sign and confidence score.
- To demonstrate the practical application of Deep Learning and Computer Vision.

---

## ✨ Features

- 🖼️ Upload ASL hand-sign images.
- 🤖 Deep Learning based image classification.
- 🔤 Supports 29 ASL classes.
- 📊 Displays prediction confidence.
- 🌐 Flask-based web application.
- 🎨 Modern and responsive user interface.
- 📥 Export prediction results as JSON.
- 📋 Copy prediction report.
- ⚡ Real-time prediction after image upload.

---

## 🔤 Supported Classes

The model supports the following 29 classes:

| No. | Class |
|---:|---|
| 0 | A |
| 1 | B |
| 2 | C |
| 3 | D |
| 4 | E |
| 5 | F |
| 6 | G |
| 7 | H |
| 8 | I |
| 9 | J |
| 10 | K |
| 11 | L |
| 12 | M |
| 13 | N |
| 14 | O |
| 15 | P |
| 16 | Q |
| 17 | R |
| 18 | S |
| 19 | T |
| 20 | U |
| 21 | V |
| 22 | W |
| 23 | X |
| 24 | Y |
| 25 | Z |
| 26 | del |
| 27 | nothing |
| 28 | space |

---

## 🧠 Technologies Used

### Programming Language

- Python

### Machine Learning

- TensorFlow
- Keras
- Convolutional Neural Network (CNN)

### Computer Vision

- Pillow (PIL)
- NumPy

### Web Development

- Flask
- HTML5
- CSS3
- JavaScript

### Development Environment

- Visual Studio Code
- Git
- GitHub

---

## 🏗️ System Architecture

The system follows the following workflow:

```text
             Input Image
                  |
                  ↓
          Image Preprocessing
                  |
                  ↓
          Resize to 64 × 64
                  |
                  ↓
          CNN Deep Learning Model
                  |
                  ↓
          Feature Extraction
                  |
                  ↓
          Image Classification
                  |
                  ↓
        Predicted ASL Class
                  |
                  ↓
       Confidence Calculation
                  |
                  ↓
          Flask Web Application
                  |
                  ↓
       Result Displayed to User