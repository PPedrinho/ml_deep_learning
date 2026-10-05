import numpy as np


def predict(
    model,
    X
):
    """
    Realiza predições sobre um conjunto de imagens.

    Returns:
        Probabilidades e classes previstas.
    """

    probabilities = model.predict(
        X,
        verbose=0
    )


    predictions = np.argmax(
        probabilities,
        axis=1
    )


    return (
        probabilities,
        predictions
    )