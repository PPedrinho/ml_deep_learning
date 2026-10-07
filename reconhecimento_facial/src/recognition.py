import numpy as np

from deepface import DeepFace
from deepface.modules.exceptions import FaceNotDetected


def cosine_distance(
    embedding_1,
    embedding_2,
):

    embedding_1 = np.asarray(
        embedding_1,
        dtype=np.float32,
    )

    embedding_2 = np.asarray(
        embedding_2,
        dtype=np.float32,
    )

    norm_1 = np.linalg.norm(
        embedding_1
    )

    norm_2 = np.linalg.norm(
        embedding_2
    )

    if norm_1 == 0 or norm_2 == 0:
        return float("inf")

    similarity = np.dot(
        embedding_1,
        embedding_2,
    ) / (
        norm_1 * norm_2
    )

    return 1 - similarity


def recognize_face(
    image_path,
    database,
    model_name,
    threshold,
):

    try:

        representation = DeepFace.represent(
            img_path=str(image_path),
            model_name=model_name,
            detector_backend="opencv",
            enforce_detection=True,
        )

    except FaceNotDetected:

        return {
            "authorized": False,
            "name": "Rosto não detectado",
            "distance": None,
            "face_detected": False,
        }

    except Exception as error:

        print(
            f"Erro ao gerar embedding de "
            f"{image_path}: {error}"
        )

        return {
            "authorized": False,
            "name": "Erro na detecção",
            "distance": None,
            "face_detected": False,
        }

    if not representation:

        return {
            "authorized": False,
            "name": "Rosto não detectado",
            "distance": None,
            "face_detected": False,
        }

    query_embedding = np.asarray(
        representation[0]["embedding"],
        dtype=np.float32,
    )

    if not database:

        return {
            "authorized": False,
            "name": "Nenhuma pessoa autorizada",
            "distance": None,
            "face_detected": True,
        }

    best_match = None

    best_distance = float("inf")


    for person in database:

        distance = cosine_distance(
            query_embedding,
            person["embedding"],
        )


        if distance < best_distance:

            best_distance = distance

            best_match = person


    if (
        best_match is not None
        and best_distance <= threshold
    ):

        return {
            "authorized": True,
            "name": best_match["name"],
            "distance": best_distance,
            "face_detected": True,
        }

    return {
        "authorized": False,
        "name": "Desconhecido",
        "distance": best_distance,
        "face_detected": True,
    }