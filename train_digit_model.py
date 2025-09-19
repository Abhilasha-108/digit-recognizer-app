# train_digit_model.py
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
# We add Dropout for better generalization
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical

# 1. Load and Preprocess the Data
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print("Data preprocessed successfully.")

# 2. Build a Deeper and More Robust CNN Model
print("Building a deeper CNN model...")
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    # Add a second convolutional layer
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    # Add a Dropout layer to prevent overfitting
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# 3. Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model compiled successfully.")

# 4. Train the Model for Longer
print("Training the model for 10 epochs...")
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10) # Increased epochs
print("Model training complete.")

# 5. Save the New, Improved Model
model.save('digit_recognizer.h5')
print("New, improved model saved to digit_recognizer.h5")