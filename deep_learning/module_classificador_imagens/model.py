from tensorflow import keras
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Input,
)


def create_model():
    """
    Cria e compila a rede neural convolucional.
    """

    model = keras.Sequential()


    # Entrada: imagem RGB 32x32
    model.add(
        Input(
            shape=(32, 32, 3)
        )
    )


    # Primeira camada convolucional
    model.add(
        Conv2D(
            filters=16,
            kernel_size=(3, 3),
            activation="relu"
        )
    )

    model.add(
        MaxPooling2D(
            (2, 2)
        )
    )


    # Segunda camada convolucional
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu"
        )
    )

    model.add(
        MaxPooling2D(
            (2, 2)
        )
    )


    # Terceira camada convolucional
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu"
        )
    )

    model.add(
        MaxPooling2D(
            (2, 2)
        )
    )


    # Camadas densas
    model.add(
        Flatten()
    )

    model.add(
        Dense(
            512,
            activation="relu"
        )
    )

    model.add(
        Dense(
            10,
            activation="softmax"
        )
    )


    # Compilação
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )


    return model