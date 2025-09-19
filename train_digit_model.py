# train_digit_model.py
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# 1. Load Data: We use the built-in MNIST dataset of 70,000 handwritten digits.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Preprocess Data: Neural networks need data in a specific, consistent format.
# We reshape images to 28x28x1 and normalize pixel values from 0-255 to 0-1.
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
# We one-hot encode the labels (e.g., 5 becomes [0,0,0,0,0,1,0,0,0,0]).
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 3. Build Model: We use a Convolutional Neural Network (CNN), which is ideal for finding patterns in images.
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax') # The final layer gives a probability for each of the 10 digits.
])

# 4. Compile Model: This sets up the model for training.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train Model: The model "learns" by looking at the images and their correct labels.
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)

# 6. Save Model: We save our trained "brain" to a file so the web app can use it.
model.save('digit_recognizer.h5')
print("Model saved as digit_recognizer.h5")