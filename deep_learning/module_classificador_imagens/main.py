from loguru import logger


from module_classificador_imagens.dataset import (
    load_data,
    describe_data,
    get_class_names,
)


from module_classificador_imagens.model import (
    create_model,
)


from module_classificador_imagens.train import (
    train_model,
)


from module_classificador_imagens.evaluate import (
    evaluate_model,
)


from module_classificador_imagens.predict import (
    predict,
)


from module_classificador_imagens.plots import (
    plot_samples,
    plot_accuracy,
    plot_loss,
    plot_predictions,
)


def main():

    logger.info(
        "Iniciando classificador de imagens"
    )


    # ==================================================
    # 1. Carregar dados
    # ==================================================

    (
        X_train,
        y_train,
        X_test,
        y_test
    ) = load_data()


    describe_data(
        X_train,
        y_train,
        X_test,
        y_test
    )


    class_names = get_class_names()


    # ==================================================
    # 2. Visualizar exemplos
    # ==================================================

    plot_samples(
        X_train,
        y_train,
        class_names
    )


    # ==================================================
    # 3. Criar modelo
    # ==================================================

    logger.info(
        "Criando modelo CNN"
    )

    model = create_model()


    model.summary()


    # ==================================================
    # 4. Treinar modelo
    # ==================================================

    logger.info(
        "Iniciando treinamento"
    )

    history = train_model(
        model,
        X_train,
        y_train
    )


    logger.success(
        "Treinamento concluído"
    )


    # ==================================================
    # 5. Visualizar treinamento
    # ==================================================

    plot_accuracy(
        history
    )

    plot_loss(
        history
    )


    # ==================================================
    # 6. Avaliar modelo
    # ==================================================

    logger.info(
        "Avaliando modelo"
    )

    evaluate_model(
        model,
        X_test,
        y_test
    )


    # ==================================================
    # 7. Realizar predições
    # ==================================================

    logger.info(
        "Realizando predições"
    )

    (
        probabilities,
        predictions
    ) = predict(
        model,
        X_test[:9]
    )


    # ==================================================
    # 8. Visualizar predições
    # ==================================================

    plot_predictions(
        X_test[:9],
        y_test[:9],
        predictions,
        class_names
    )


    logger.success(
        "Pipeline executado com sucesso"
    )


if __name__ == "__main__":
    main()