from pathlib import Path

import numpy as np

from deepface import DeepFace


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def load_authorized_faces(
    authorized_dir: Path,
    model_name: str,
):

    database = []


    for person_dir in authorized_dir.iterdir():

        if not person_dir.is_dir():
            continue


        person_name = person_dir.name


        for image_path in person_dir.iterdir():

            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue


            try:

                embedding_result = DeepFace.represent(
                    img_path=str(image_path),
                    model_name=model_name,
                    detector_backend="opencv",
                    enforce_detection=True,
                )


                embedding = np.asarray(
                    embedding_result[0]["embedding"],
                    dtype=np.float32,
                )


                database.append(
                    {
                        "name": person_name,
                        "embedding": embedding,
                        "image": image_path,
                    }
                )


                print(
                    f"Embedding criado: "
                    f"{person_name} - {image_path.name}"
                )


            except Exception as error:

                print(
                    f"Erro ao processar "
                    f"{image_path}: {error}"
                )


    return database