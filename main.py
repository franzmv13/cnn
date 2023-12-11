from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img
import cv2 , os
import numpy as np

import shutil


# Specify the directory path to clear
directory = "D:\school\comscie\emotion.V2\\temp"

# Call the function to clear the directory





def prediction(imgs,model):
    model = load_model(model)
    try: 
        clear_directory(directory)
        path = faces(imgs)
        

        img = image.load_img(path, target_size=(48, 48))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize pixel values

        # Make predictions
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        emotion_classes = ['angry',  'happy', 'neutral', 'sad']
        return emotion_classes[predicted_class]
    except Exception as error:
        img = image.load_img(imgs, target_size=(48, 48))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize pixel values

        # Make predictions
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        emotion_classes = ['angry',  'happy', 'neutral', 'sad']
        return emotion_classes[predicted_class]

def clear_directory(directory_path):
    try:
        # Delete the directory and its contents
        shutil.rmtree(directory_path)
        # Recreate an empty directory
        os.makedirs(directory_path)

      
    except Exception as e:
        print(f"Error clearing directory {directory_path}: {e}")


def faces(image_path):
    
    output_folder = r"D:\school\comscie\emotion.V2\temp"

    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the input image
    image = cv2.imread(image_path)

    # Check if the image is successfully loaded
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=2.4, minNeighbors=5)

    # Crop and save each detected face
    for i, (x, y, w, h) in enumerate(faces):
        cropped_face = image[y:y+h, x:x+w]
        output_path = f"{output_folder}/face_{i + 1}.jpg"
        cv2.imwrite(output_path, cropped_face)
        return output_path
        

