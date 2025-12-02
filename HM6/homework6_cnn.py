import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load and preprocess data
(train_imgs, train_labels), (test_imgs, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to [0,1]
train_imgs = train_imgs.astype('float32') / 255.0
test_imgs = test_imgs.astype('float32') / 255.0

# Add channel dimension (batch, height, width, channels)
train_imgs = np.expand_dims(train_imgs, -1)  # shape = (60000, 28, 28, 1)
test_imgs  = np.expand_dims(test_imgs, -1)

# (Optional) convert labels to one-hot encoding
train_labels_cat = to_categorical(train_labels, num_classes=10)
test_labels_cat  = to_categorical(test_labels,  num_classes=10)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Training
history = model.fit(train_imgs, train_labels_cat,
                    epochs=10,
                    batch_size=64,
                    validation_split=0.1)

test_loss, test_acc = model.evaluate(test_imgs, test_labels_cat, verbose=2)
print("Test accuracy:", test_acc)

predictions = model.predict(test_imgs[:10])
pred_classes = np.argmax(predictions, axis=1)
print("Predicted:", pred_classes)
print("Ground truth:", test_labels[:10])
