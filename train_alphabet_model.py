# train_alphabet_model.py
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical

# 1. Load the EMNIST "letters" dataset
print("Loading EMNIST/letters dataset...")
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# 2. Preprocess the data
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    # EMNIST images are rotated and flipped, we need to fix them
    image = tf.image.rot90(image, k=3)
    image = tf.image.flip_left_right(image)
    # The labels are 1-26, but neural networks need 0-25.
    label = label - 1
    return image, label

ds_train = ds_train.map(preprocess).batch(128)
ds_test = ds_test.map(preprocess).batch(128)
print("Data preprocessed successfully.")

# 3. Build the CNN Model (Same as before, but the final layer is different)
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    # CRUCIAL CHANGE: The output layer now has 26 neurons for 26 letters.
    Dense(26, activation='softmax')
])

# 4. Compile and Train the Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Training the alphabet model...")
model.fit(ds_train, epochs=10, validation_data=ds_test)
print("Model training complete.")

# 5. Save the New Alphabet Model
model.save('alphabet_recognizer.h5')
print("Alphabet model saved to alphabet_recognizer.h5")