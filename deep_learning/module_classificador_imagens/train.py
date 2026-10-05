from module_classificador_imagens.config import (
    EPOCHS,
    BATCH_SIZE,
    VALIDATION_SPLIT,
)


def train_model(
    model,
    X_train,
    y_train
):
    """
    Treina o modelo utilizando os dados de treinamento.
    """

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
    )


    return history