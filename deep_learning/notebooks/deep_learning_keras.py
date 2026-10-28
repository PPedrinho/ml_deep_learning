import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


SEED = 42
EPOCHS = 15
BATCH_SIZE = 128
RESULTS_DIR = "reports"


def load_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    return x_train, y_train, x_test, y_test


def build_model():
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28, 1)),
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.08),
            keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.35),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_training_curve(history):
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="treino")
    axes[0].plot(history.history["val_loss"], label="validação")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Época")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="treino")
    axes[1].plot(history.history["val_accuracy"], label="validação")
    axes[1].set_title("Acurácia")
    axes[1].set_xlabel("Época")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(
        Path(RESULTS_DIR) / "keras_training.png",
        dpi=150,
    )
    plt.close(figure)


def save_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix)
    figure.colorbar(image, ax=axis)

    axis.set_xlabel("Classe prevista")
    axis.set_ylabel("Classe real")
    axis.set_title("Matriz de confusão")

    figure.tight_layout()
    figure.savefig(
        Path(RESULTS_DIR) / "keras_confusion_matrix.png",
        dpi=150,
    )
    plt.close(figure)


def main():
    keras.utils.set_random_seed(SEED)

    print(f"Keras: {keras.__version__}")
    print("Carregando Fashion-MNIST...")

    x_train, y_train, x_test, y_test = load_dataset()

    model = build_model()

    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    loss, accuracy = model.evaluate(
        x_test,
        y_test,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    probabilities = model.predict(
        x_test,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    predictions = np.argmax(probabilities, axis=1)

    print(f"\nLoss no teste: {loss:.4f}")
    print(f"Acurácia no teste: {accuracy:.4f}")
    print("\nRelatório de classificação:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
        )
    )

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    model.save(
        Path(RESULTS_DIR) / "fashion_mnist.keras"
    )

    save_training_curve(history)
    save_confusion_matrix(y_test, predictions)

    print("\nArquivos gerados:")
    print(f"- {RESULTS_DIR}/fashion_mnist.keras")
    print(f"- {RESULTS_DIR}/keras_training.png")
    print(f"- {RESULTS_DIR}/keras_confusion_matrix.png")


if __name__ == "__main__":
    main()
