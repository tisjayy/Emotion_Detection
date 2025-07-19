import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Activation,
                                     GlobalAveragePooling2D, Dense, Add, Input)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import mlflow.tensorflow

# ------------- Model Blocks -------------
def residual_block(x, filters, downsample=False):
    stride = 2 if downsample else 1
    shortcut = x
    x = Conv2D(filters, (3, 3), strides=stride, padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    if downsample or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_resnet(input_shape=(48, 48, 1), num_classes=7, n=5):
    inputs = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    for _ in range(n):
        x = residual_block(x, 16)
    x = residual_block(x, 32, downsample=True)
    for _ in range(n - 1):
        x = residual_block(x, 32)
    x = residual_block(x, 64, downsample=True)
    for _ in range(n - 1):
        x = residual_block(x, 64)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

# ------------- MLflow Callback -------------
class MLflowMetrics(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            mlflow.log_metric(k, float(v), step=epoch)

if __name__ == "__main__":
    # ---------------- Data ----------------
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')  # unlabeled

    y = train_df['emotion'].values
    X = np.array([np.fromstring(p, sep=' ') for p in train_df['pixels']])
    X_test = np.array([np.fromstring(p, sep=' ') for p in test_df['pixels']])

    X = X.reshape(-1, 48, 48, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 48, 48, 1).astype('float32') / 255.0
    y = to_categorical(y, 7)

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y.argmax(axis=1)
    )

    datagen = ImageDataGenerator(horizontal_flip=True)
    datagen.fit(X_train)

    # ---------------- Hyperparams ----------------
    epochs = 40
    batch_size = 128
    lr = 0.1
    depth_n = 5

    model = build_resnet(n=depth_n)
    optimizer = SGD(learning_rate=lr, momentum=0.9, decay=1e-4)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    callbacks = [
        MLflowMetrics(),
        EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, min_lr=1e-4)
    ]

    # ---------------- MLflow Run ----------------
    mlflow.set_experiment("emotion_recognition")
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate_init", lr)
        mlflow.log_param("architecture", f"CustomResNet_n{depth_n}")
        mlflow.log_param("augmentation", "horizontal_flip")
        mlflow.log_param("l2_weight_decay", 1e-4)

        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            verbose=1,
            callbacks=callbacks
        )

        # Final validation metrics
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        mlflow.log_metric("val_final_loss", float(val_loss))
        mlflow.log_metric("val_final_accuracy", float(val_acc))

        # Save & log model
        model.save("model_resnet_emotion.h5")
        mlflow.log_artifact("model_resnet_emotion.h5")

