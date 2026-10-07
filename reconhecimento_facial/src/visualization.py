import cv2

from src.config import (
    GREEN,
    RED,
    WHITE,
)


def draw_result(
    image,
    face,
    authorized,
    name,
    distance,
):
 

    x = face["facial_area"]["x"]
    y = face["facial_area"]["y"]

    w = face["facial_area"]["w"]
    h = face["facial_area"]["h"]


    if authorized:

        color = GREEN

        label = (
            f"ACESSO LIBERADO - {name}"
        )

    else:

        color = RED

        label = "ACESSO NEGADO"


    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        color,
        3,
    )


    cv2.rectangle(
        image,
        (x, y - 35),
        (x + w, y),
        color,
        -1,
    )


    cv2.putText(
        image,
        label,
        (x + 5, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        WHITE,
        2,
        cv2.LINE_AA,
    )


    
    cv2.putText(
        image,
        f"dist: {distance:.3f}",
        (x, y + h + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


    return image