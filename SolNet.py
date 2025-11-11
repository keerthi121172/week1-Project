import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

DATASET_DIR = r"C:\Users\keert\Downloads\Detect_Solar_Dust"
IMG_SIZE = (227, 227)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001
SEED = 42
MODEL_SAVE_PATH = r"C:\Users\keert\OneDrive\Desktop\solnet_model.keras"

def build_solnet(input_shape=(227, 227, 3)):
    model = models.Sequential(name="SolNet")

    model.add(layers.Conv2D(64, (11, 11), strides=4, padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (5, 5), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (2, 2), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(256, (2, 2), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation='sigmoid'))  

    return model

def load_dataset():
    data = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=True,
        seed=SEED,
        validation_split=0.2,
        subset='both'
    )
    train_ds, val_ds = data
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds

def train_and_evaluate():
    train_ds, val_ds = load_dataset()

    model = build_solnet()
    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    cb = [
        callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb)
    model.save(MODEL_SAVE_PATH)

    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend((preds > 0.5).astype(int).ravel())

    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend(); plt.title("Accuracy")

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend(); plt.title("Loss")
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()
