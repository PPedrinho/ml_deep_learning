import matplotlib.pyplot as plt


def plot_samples(
    X_train,
    y_train,
    class_names
):
    """
    Exibe exemplos do conjunto de treinamento.
    """

    plt.figure(
        figsize=(10, 10)
    )


    for i in range(9):

        plt.subplot(
            3,
            3,
            i + 1
        )

        plt.imshow(
            X_train[i]
        )

        plt.title(
            class_names[y_train[i]]
        )

        plt.axis("off")


    plt.suptitle(
        "Exemplos da base CIFAR-10"
    )

    plt.tight_layout()

    plt.show()


def plot_accuracy(history):
    """
    Plota a acurácia de treinamento e validação.
    """

    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        history.history["accuracy"],
        label="Treino"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validação"
    )


    plt.title(
        "Acurácia por época"
    )

    plt.xlabel(
        "Épocas"
    )

    plt.ylabel(
        "Acurácia"
    )

    plt.legend()

    plt.show()


def plot_loss(history):
    """
    Plota a loss de treinamento e validação.
    """

    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        history.history["loss"],
        label="Treino"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validação"
    )


    plt.title(
        "Loss por época"
    )

    plt.xlabel(
        "Épocas"
    )

    plt.ylabel(
        "Loss"
    )

    plt.legend()

    plt.show()


def plot_predictions(
    X_test,
    y_test,
    predictions,
    class_names
):
    """
    Exibe imagens do conjunto de teste
    com seus rótulos reais e previstos.
    """

    plt.figure(
        figsize=(10, 10)
    )


    for i in range(9):

        plt.subplot(
            3,
            3,
            i + 1
        )

        plt.imshow(
            X_test[i]
        )


        real = class_names[
            y_test[i]
        ]

        predicted = class_names[
            predictions[i]
        ]


        plt.title(
            f"Real: {real}\n"
            f"Predição: {predicted}"
        )

        plt.axis("off")


    plt.suptitle(
        "Previsões em imagens "
        "do conjunto de teste"
    )

    plt.tight_layout()

    plt.show()