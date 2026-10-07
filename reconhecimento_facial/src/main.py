import cv2

from loguru import logger

from deepface import DeepFace


from src.config import (
    AUTHORIZED_DIR,
    TEST_DIR,
    RESULTS_DIR,
    MODEL_NAME,
    THRESHOLD,
)


from src.database import (
    load_authorized_faces,
)


from src.recognition import (
    recognize_face,
)


def process_image(
    image_path,
    database,
):


    logger.info(
        f"Processando: {image_path.name}"
    )


    image = cv2.imread(
        str(image_path)
    )


    if image is None:

        logger.error(
            f"Não foi possível abrir: "
            f"{image_path}"
        )

        return



    try:

        faces = DeepFace.extract_faces(
            img_path=str(image_path),
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
        )


    except Exception as error:

        logger.error(
            f"Erro na detecção facial: "
            f"{error}"
        )

        return


  

    if not faces:

        logger.warning(
            f"Nenhum rosto encontrado em "
            f"{image_path.name}"
        )

        label = "ROSTO NÃO DETECTADO"

        color = (0, 0, 255)


        height, width = image.shape[:2]


        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            2,
        )[0]


        text_x = (
            width - text_size[0]
        ) // 2


        text_y = (
            height + text_size[1]
        ) // 2


        cv2.putText(
            image,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )


        output_path = (
            RESULTS_DIR
            / image_path.name
        )


        cv2.imwrite(
            str(output_path),
            image,
        )


        logger.success(
            f"Resultado salvo em: "
            f"{output_path}"
        )

        return



    for face in faces:

        facial_area = face[
            "facial_area"
        ]



        x = facial_area["x"]
        y = facial_area["y"]
        w = facial_area["w"]
        h = facial_area["h"]


        result = recognize_face(
            image_path=image_path,
            database=database,
            model_name=MODEL_NAME,
            threshold=THRESHOLD,
        )



        if not result["face_detected"]:

            color = (0, 0, 255)

            label = "ROSTO NÃO DETECTADO"


        elif result["authorized"]:

            color = (0, 255, 0)

            label = (
                f"ACESSO LIBERADO - "
                f"{result['name']}"
            )


        else:

            color = (0, 0, 255)

            label = "ACESSO NEGADO"



        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            color,
            3,
        )


        text_y = max(
            y - 10,
            25,
        )



        cv2.putText(
            image,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


        if result["distance"] is not None:

            logger.info(
                f"{image_path.name}: "
                f"{label} | "
                f"distância="
                f"{result['distance']:.4f}"
            )

        else:

            logger.info(
                f"{image_path.name}: "
                f"{label}"
            )


    output_path = (
        RESULTS_DIR
        / image_path.name
    )


    cv2.imwrite(
        str(output_path),
        image,
    )


    logger.success(
        f"Resultado salvo em: "
        f"{output_path}"
    )


def main():

    logger.info(
        "Iniciando sistema de reconhecimento facial"
    )


    logger.info(
        "Carregando pessoas autorizadas..."
    )


    database = load_authorized_faces(
        authorized_dir=AUTHORIZED_DIR,
        model_name=MODEL_NAME,
    )


    logger.success(
        f"{len(database)} imagens autorizadas carregadas."
    )


    test_images = [
        path
        for path in TEST_DIR.iterdir()
        if path.suffix.lower()
        in {
            ".jpg",
            ".jpeg",
            ".png",
        }
    ]


    logger.info(
        f"{len(test_images)} imagens de teste encontradas."
    )


    for image_path in test_images:

        process_image(
            image_path=image_path,
            database=database,
        )


    logger.success(
        "Processamento finalizado."
    )


if __name__ == "__main__":
    main()