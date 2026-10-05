import numpy as np

from tensorflow import keras

from module_classificador_imagens.config import CLASS_NAMES


def load_data():
    """
    Carrega e prepara o dataset CIFAR-10.

    Returns:
        X_train: imagens de treinamento.
        y_train: rótulos de treinamento.
        X_test: imagens de teste.
        y_test: rótulos de teste.
    """

    (
        X_train,
        y_train
    ), (
        X_test,
        y_test
    ) = keras.datasets.cifar10.load_data()


    # Normalização dos pixels para o intervalo [0, 1]
    X_train = X_train.astype(
        "float32"
    ) / 255.0

    X_test = X_test.astype(
        "float32"
    ) / 255.0


    # Converte os rótulos de (n, 1) para (n,)
    y_train = y_train.flatten()
    y_test = y_test.flatten()


    return (
        X_train,
        y_train,
        X_test,
        y_test
    )


def get_class_names():
    """
    Retorna os nomes das classes do CIFAR-10.
    """

    return CLASS_NAMES


def describe_data(
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Exibe informações básicas sobre os conjuntos de dados.
    """

    print(
        f"Formato de X_train: {X_train.shape}"
    )

    print(
        f"Formato de y_train: {y_train.shape}"
    )

    print(
        f"Formato de X_test: {X_test.shape}"
    )

    print(
        f"Formato de y_test: {y_test.shape}"
    )