# train_digit_model.py
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
# Import the ImageDataGenerator for Data Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Load and Preprocess the Data
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print("Data preprocessed successfully.")

# 2. Build the same deep CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# 3. Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model compiled successfully.")

# 4. Create the Data Augmentation Generator
# This will create new images on the fly with random transformations.
datagen = ImageDataGenerator(
    rotation_range=10,      # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,         # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally
    height_shift_range=0.1  # randomly shift images vertically
)
datagen.fit(x_train)

# 5. Train the Model using Augmented Data
# We use datagen.flow() to feed the augmented images to the model.
print("Training the model with data augmentation...")
model.fit(datagen.flow(x_train, y_train, batch_size=32),
          validation_data=(x_test, y_test),
          epochs=15) # Increased epochs for more thorough training
print("Model training complete.")

# 6. Save the Final, Most Robust Model
model.save('digit_recognizer.h5')
print("Final, augmented model saved to digit_recognizer.h5")