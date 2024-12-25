Emotion Detection Using Deep Learning
Introduction
This project aims to classify the emotion on a person's face into one of seven categories using deep convolutional neural networks. The model is trained on the FER-2013 dataset, which was published at the International Conference on Machine Learning (ICML). This dataset consists of 35,887 grayscale, 48x48-sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad, and surprised.

Dependencies
To run this project, you need the following:

Python 3
OpenCV
TensorFlow
Install Dependencies
To install the required Python packages, create a virtual environment, activate it, and install the dependencies by running:

bash
Copy code
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# For Windows
venv\Scripts\activate

# For macOS/Linux
source venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt
Basic Usage
This project is compatible with tensorflow-2.0 and makes use of the Keras API through tensorflow.keras.

Clone the Repository and Setup
bash
Copy code
git clone https://github.com/atulapra/Emotion-detection.git
cd Emotion-detection
Download Required Files
Before running the project, you need the following files:

Haar Cascade Files:

Download haarcascade_frontalface_default.xml
Download haarcascade_eye.xml
Place these files inside the src folder. These files are used for detecting faces and eyes from the webcam feed.

Pre-Trained Model:

Download the pre-trained model (model.h5)
Place the model.h5 file inside the src folder to use it for emotion detection without retraining the model.

Data Preparation
Download the FER-2013 dataset and place it inside the src folder. You can download it from the Kaggle FER-2013 Dataset.

Alternatively, if you want to experiment with your own dataset in CSV format, use the code in dataset_prepare.py for preprocessing.

Training the Model
If you want to train the model from scratch, run:

bash
Copy code
cd src
python emotions.py --mode train
This will start the training process using the FER-2013 dataset.

Using the Pre-Trained Model
If you prefer to use the pre-trained model, after downloading the model.h5 file, place it in the src folder.

Running the Script for Emotion Detection
To run the emotion detection using the webcam feed, after placing the pre-trained model.h5 file in the src folder, use the following command:

bash
Copy code
cd src
python emotions.py --mode display
This will launch the webcam and start detecting emotions on all faces in the webcam feed.

Folder Structure
The folder structure should look like this:

bash
Copy code
Emotion-detection/
│
└── src/
    ├── data/                # Folder for dataset
    ├── emotions.py          # Main script for training or displaying emotions
    ├── haarcascade_frontalface_default.xml  # Haar Cascade for face detection
    ├── haarcascade_eye.xml  # Haar Cascade for eye detection
    ├── model.h5             # Pre-trained model
    ├── dataset_prepare.py    # Code for dataset preprocessing (optional)
    └── minfile.py            # Minified script (optional for advanced use)
Running the Minified Script (minfile.py)
If you prefer a more compact version of the code or want to integrate it into a larger system, you can run the minfile.py for quick execution. To do this, use:

bash
Copy code
cd src
python minfile.py
Algorithm
Face Detection: The Haar Cascade method is used to detect faces in each frame from the webcam feed.
Emotion Classification: The region of the image containing the face is resized to 48x48 pixels and passed as input to the Convolutional Neural Network (CNN).
Softmax Output: The CNN outputs a list of softmax scores for the seven emotion classes.
Display Emotion: The emotion with the highest score is displayed on the screen.
Accuracy
With a simple 4-layer CNN, the test accuracy reached 63.2% after 50 epochs.
![alt text](figure/Figure_1.png)



References
FER-2013 Dataset: https://www.kaggle.com/deadskull7/fer2013
Challenges in Representation Learning: A report on three machine learning contests. I Goodfellow, D Erhan, PL Carrier, A Courville, et al. arXiv 2013.
Important Notes
Make sure to download the pre-trained model model.h5 and place it in the src folder if you are not training the model.
Ensure you have the Haar Cascade files (haarcascade_frontalface_default.xml and haarcascade_eye.xml) in the src folder for face and eye detection.
Troubleshooting
If the webcam feed is not working or faces are not detected, ensure your camera drivers are up-to-date and the webcam is functional.
If you encounter errors related to TensorFlow, check your Python version and ensure compatibility with TensorFlow 2.0.
