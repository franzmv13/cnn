from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img

import numpy as np
import os

# Load the trained model
model = load_model('emotion_model20.h5')
# Folder containing test images
test_folder = 'D:\school\comscie\emotion.V2\\test_pics\happy\\'
falsepredict=[] #store false predictions
# List to store predictions
predictions = []
# Iterate over each image in the test folder
for filename in os.listdir(test_folder):
    img_path = os.path.join(test_folder, filename)

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values

    # Make predictions
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    emotion_classes = ['angry',  'happy', 'neutral', 'sad']
    predicted_emotion = emotion_classes[predicted_class]


    predictions.append((predicted_emotion))

print(predictions)
#
# for filename, emotion in predictions:
#     print(f'{filename}: {emotion}')
