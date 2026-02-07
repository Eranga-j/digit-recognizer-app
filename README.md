🔢 Handwritten Digit Recognizer App

This is a Handwritten Digit Recognition application built using Python, TensorFlow (CNN), and Gradio.
It allows users to draw a digit (0–9) or upload an image, and the app predicts the digit using a trained CNN model.

🚀 Features

Draw a digit using a canvas

Upload a single digit image

Automatic image preprocessing (resize, normalize)

Shows:

Predicted digit

Confidence score

Probability distribution for all digits (0–9)

One-click run using a .bat file (Windows)

📂 Project Structure
digit-recognizer-app/
│
├── .venv/                          # Virtual environment
├── digit_cnn_model_fixed_keras215.h5  # Trained CNN model
├── gradio_app.py                   # Main application
├── run_app.bat                     # Auto-run file (Windows)
├── README.md                       # Project documentation

🛠 Requirements

Windows OS

Python 3.10

Internet connection (first run only)

Libraries used:

tensorflow

numpy

pillow

gradio

(All installed inside .venv)

▶ How to Run the App (Easiest Way)
✅ Method 1: Double-Click (Recommended)

Open the folder digit-recognizer-app

Double-click run_app.bat

Wait a few seconds

Browser opens automatically at:

http://127.0.0.1:7860

🖥 Method 2: Command Line (Optional)
cd digit-recognizer-app
.\.venv\Scripts\activate
python gradio_app.py

✍ How to Use

Draw ONE digit only (0–9) in the canvas
OR

Upload an image with a single digit

Click Predict

View:

Predicted digit

Confidence

Probability chart

⚠ Draw only one digit for best accuracy.

📘 Model Information

Model type: Convolutional Neural Network (CNN)

Input size: 28 × 28 grayscale

Dataset style: MNIST-like

Output: 10 classes (digits 0–9)

❗ Notes

Drawing multiple separated strokes may reduce accuracy

Clear the canvas before drawing a new digit

Best results when digit is centered and bold

👨‍🎓 Use Case

AI / ML assignments

CNN demonstrations

Image classification learning

Educational projects

