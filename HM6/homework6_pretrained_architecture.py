import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Load CIFAR-100
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
num_classes = 100

# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test, num_classes)

# Fast preprocessing: resize to 96x96 for MobileNetV2
def preprocess_img(img, label):
    img = tf.image.resize(img, (96, 96))
    img = tf.cast(img, tf.float32)
    img = keras.applications.mobilenet_v2.preprocess_input(img)
    return img, label

batch_size = 64
AUTOTUNE = tf.data.AUTOTUNE

# Training dataset
train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(3000)
    .map(preprocess_img, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

# Validation dataset
val_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .map(preprocess_img, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

# Pretrained MobileNetV2 (very fast)
base_model = keras.applications.MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_shape=(96, 96, 3)
)
base_model.trainable = False  # keep pretrained layers frozen

# Build complete model
inputs = keras.Input(shape=(96, 96, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)  # optional
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = keras.Model(inputs, outputs)

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train ONLY 3 epochs (fast)
model.fit(
    train_ds,
    epochs=3,
    validation_data=val_ds
)

# Evaluate
loss, acc = model.evaluate(val_ds)
print("Test accuracy:", acc)
