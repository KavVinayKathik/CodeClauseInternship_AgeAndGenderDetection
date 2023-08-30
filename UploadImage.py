import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
import cv2
from keras.models import load_model
import time

# Load the trained model
model = load_model('/Users/kavvinaykarthik/Desktop/FaceRecognite/age_gender_model2.h5')  # Replace with the path to your saved model

# Gender labels
gender_dict = {0: 'Male', 1: 'Female'}

# Load and preprocess the image
def preprocess_image(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (128, 128))
    frame_array = frame / 255.0
    frame_array = np.expand_dims(frame_array, axis=0)
    return frame_array

# Ask the user to choose image source
print("Select image source:")
print("1. Webcam")
print("2. Image Upload")
choice = int(input("Enter your choice (1/2): "))

if choice == 1:
    # Open the webcam
    cap = cv2.VideoCapture(0)

    start_time = None
    captured_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Use a face detection model to detect faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            if start_time is None:
                start_time = time.time()

            captured_frames.append(frame)

            # Display a rectangle around the detected face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if start_time is not None and time.time() - start_time >= 3:  # Adjust the time threshold as needed
            break

        # Display the frame
        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

    # Process the captured face frames
    highest_probability = 0
    best_frame = None

    for frame in captured_frames:
        frame_array = preprocess_image(frame)

        # Perform gender and age prediction
        pred_gender, pred_age = model.predict(frame_array)

        # Calculate the probability of the predicted gender
        prob_gender = max(pred_gender[0][0], 1 - pred_gender[0][0])  # Take the maximum of the gender probabilities

        if prob_gender > highest_probability:
            highest_probability = prob_gender
            best_frame = frame

    # If a frame with higher probability was found, display the results
    if best_frame is not None:
        frame_array = preprocess_image(best_frame)

        # Perform gender and age prediction
        pred_gender, pred_age = model.predict(frame_array)

        # Convert gender prediction to label
        predicted_gender = gender_dict[round(pred_gender[0][0])]

        # Determine age range based on prediction
        age = round(pred_age[0][0])
        if age < 18:
            age_range = "Child (0-17)"
        elif age < 35:
            age_range = "Young Adult (18-34)"
        elif age < 50:
            age_range = "Adult (35-49)"
        else:
            age_range = "Senior (50+)"

        # Display results on the frame
        label = f"Predicted Gender: {predicted_gender}, Predicted Age: {age_range}, approx: {age}"
        cv2.putText(best_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with results
        cv2.imshow('Age and Gender Detection', best_frame)
        cv2.waitKey(0)

    # Release the window
    cv2.destroyAllWindows()

elif choice == 2:
    # Ask for the path to the input image
    input_image_path = input("Enter the path to the input image: ")

    # Load and preprocess the input image
    input_image = preprocess_image(cv2.imread(input_image_path))

    # Perform age and gender prediction
    pred_gender, pred_age = model.predict(input_image)

    # Convert gender prediction to label
    predicted_gender = gender_dict[round(pred_gender[0][0])]

    # Determine age range based on prediction
    age = round(pred_age[0][0])
    if age < 18:
        age_range = "Child (0-17)"
    elif age < 35:
        age_range = "Young Adult (18-34)"
    elif age < 50:
        age_range = "Adult (35-49)"
    else:
        age_range = "Senior (50+)"

    # Display results
    print("Predicted Gender:", predicted_gender)
    print("Predicted Age Range:", age_range)
    print("Predicted Age:", age)

else:
    print("Invalid choice. Please choose 1 or 2.")
