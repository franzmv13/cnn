import cv2

# def crop_face(image_path, output_path):
#     # Load the pre-trained face detection model
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     # Read the input image
#     image = cv2.imread(image_path)

#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the image
#     faces = face_cascade.detectMultiScale(gray_image, scaleFactor=2.4, minNeighbors=5 )

#     # If faces are found, crop the first face
#     if len(faces) > 0:
#         x, y, w, h = faces[0]
#         cropped_face = image[y:y+h, x:x+w]

#         # Save the cropped face to the output path
#         cv2.imwrite(output_path, cropped_face)
#         print("Face cropped and saved successfully.")
#     else:
#         print("No face found in the image.")

# # Specify the input and output paths
# input_image_path = r"D:\school\comscie\emotion.V2\test_pics\happy\2.jpg"
# output_image_path = "temp\pic.jpg"

# # Call the function to crop the face
# crop_face(input_image_path, output_image_path)









# import cv2

# def crop_all_faces(image_path, output_folder):
#     # Load the pre-trained face detection model
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     # Read the input image
#     image = cv2.imread(image_path)

#     # Check if the image is successfully loaded
#     if image is None:
#         print(f"Error: Unable to load image from {image_path}")
#         return

#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the image
#     faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

#     # Crop and save each detected face
#     for i, (x, y, w, h) in enumerate(faces):
#         cropped_face = image[y:y+h, x:x+w]
#         output_path = f"{output_folder}/face_{i + 1}.jpg"
#         cv2.imwrite(output_path, cropped_face)
#         print(f"Face {i + 1} cropped and saved successfully.")

# # Specify the input image and output folder
# input_image_path = r"D:\school\\comscie\\emotion.V2\\test_pics\\happy\\attractive-caucasian-man-smiling-and-showing-thumb-up-gesture-with-both-hands-over-yellow.jpg"
# output_folder = r"D:\school\comscie\emotion.V2\temp"

# # Call the function to crop all faces
# crop_all_faces(input_image_path, output_folder)


import cv2

def crop_face(image_path, output_path):
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the input image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=2, minNeighbors=5)

    # If faces are found, crop the first face and make it bigger
    if len(faces) > 0:
        x, y, w, h = faces[0]
        # Expand the cropping area to make the face bigger
        padding = 20
        x -= padding
        y -= padding
        w += 2 * padding
        h += 2 * padding
        cropped_face = image[max(0, y):min(y+h, image.shape[0]), max(0, x):min(x+w, image.shape[1])]

        # Save the cropped face to the output path
        cv2.imwrite(output_path, cropped_face)
        print("Face cropped and saved successfully.")
    else:
        print("No face found in the image.")

# Specify the input and output paths
input_image_path = r"D:\school\\comscie\\emotion.V2\\test_pics\\happy\\3.png"
output_image_path = "temp\pic.jpg"

# Call the function to crop the face
crop_face(input_image_path, output_image_path)
