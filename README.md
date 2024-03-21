# Hand Gesture Recognition for Navigation

This project utilizes hand gesture recognition to interact with a virtual environment, specifically navigating through a building's floor plan. It captures hand gestures via a webcam, recognizes them through a trained machine learning model, and visualizes paths on a floor plan based on the recognized gestures.

## Prerequisites

To run this project, you will need:

- Python 3.6 or newer
- OpenCV (`cv2`): For image processing and webcam interactions.
- MediaPipe: For hand tracking and gesture recognition.
- scikit-learn: For loading and using the machine learning model.
- NumPy: For numerical operations.
- Predefined floor plan images: These should be named according to their floor numbers and placed in a directory named `floor-plan`.

## Using the System
- Use collect_imgs.py to create your own image collections, or put your images in a folder named `data`.

- Run create_dataset.py to process the set of images to extract and normalize hand landmark data using MediaPipe and save the dataset.
  
- Run train_classifier.py for loading and preparing data, splitting it into training and testing sets, training a model, evaluating its performance, and saving the trained model
  
- **Gesture Recognition for Input:** Present your hand gestures to the webcam to input the desired floor number.
  
- **Visualize Path:** Once a floor number is inputted and confirmed with the 'ok' gesture, the system will visualize the path to the specified room on the floor plan.

