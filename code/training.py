import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Set your data directories
train_data_dir = 'train' #main ds
validation_data_dir = 'train'
batch_size = 32
epochs = 5

# Step 2: Preprocess Your Data
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(     
    train_data_dir,
    target_size=(48, 48), #img size
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(48, 48),#img size
    batch_size=batch_size,
    class_mode='categorical'
)

# Step 3: Build a Convolutional Neural Network (CNN) bobo haha
model = models.Sequential()
""" Type: Convolutional layer (Conv2D)
Number of Filters (Kernels): 32
Filter (Kernel) Size: (3, 3)
Activation Function: ReLU (Rectified Linear Unit)
Input Shape: (48, 48, 3)
Explanation: This layer applies 32 convolutional filters"""
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))  # Adjusted for 4 classes

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Train the Model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator  # Pass the validation generator here
)

# Step 5: Save the Model
model.save('test.h5')
