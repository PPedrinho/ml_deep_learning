def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Avalia o modelo no conjunto de teste.
    """

    test_loss, test_acc = model.evaluate(
        X_test,
        y_test,
        verbose=1
    )


    print(
        f"Acurácia no conjunto de teste: "
        f"{test_acc:.2f}"
    )

    print(
        f"Loss no teste: "
        f"{test_loss:.4f}"
    )


    return (
        test_loss,
        test_acc
    )