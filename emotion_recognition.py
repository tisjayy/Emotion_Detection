import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Add, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Model
import pandas as pd

# --- Residual Block ---
def residual_block(x, filters, downsample=False):
    stride = 2 if downsample else 1

    shortcut = x

    x = Conv2D(filters, (3, 3), strides=stride, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)

    # Match shortcut shape if downsampling
    if downsample or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# --- Build ResNet ---
def build_resnet(input_shape=(48, 48, 1), num_classes=7, n=5):
    inputs = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Residual blocks
    for _ in range(n):
        x = residual_block(x, 16)
    x = residual_block(x, 32, downsample=True)
    for _ in range(n-1):
        x = residual_block(x, 32)
    x = residual_block(x, 64, downsample=True)
    for _ in range(n-1):
        x = residual_block(x, 64)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv('data/train.csv')

    test_df = pd.read_csv('data/test.csv')

    Y = train_df['emotion'].values


    X = np.array([np.fromstring(pixels, sep=' ') for pixels in train_df['pixels']])
    X_test = np.array([np.fromstring(pixels, sep=' ') for pixels in test_df['pixels']])

    # Reshape and normalize
    X = X.reshape(-1, 48, 48, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 48, 48, 1).astype('float32') / 255.0

    # One-hot encode labels
    Y = to_categorical(Y, 7)

    # Image augmentation
    datagen = ImageDataGenerator(horizontal_flip=True)
    datagen.fit(X)

    # Build model
    model = build_resnet()
    model.compile(optimizer=SGD(learning_rate=0.1, momentum=0.9, decay=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train
    model.fit(datagen.flow(X, Y, batch_size=128),
             
              epochs=40,
              verbose=1)

    # Evaluate
 

    # Save model
    model.save('model_resnet_emotion.h5')

    # Predict test image (optional)
    # predict_value = np.loadtxt('test_image.csv', delimiter=',').reshape(1, 48, 48, 1).astype('float32') / 255.0
    # prediction = model.predict(predict_value)
    # print("Prediction:", prediction)
