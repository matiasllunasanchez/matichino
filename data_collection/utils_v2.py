from scipy.spatial import distance as dist
from PIL import Image
import imageio.v2 as imageio
import json
import os
import statistics
from constants import (
    OUTPUTS_PATH,
)

OFFSET_BOTTOM_SQUARE = 80


def drawFollowerRectanglev2(frame, cv2, x1, y1, x2, y2):
    ##  CUADRADO PARA DETECTAR UBICACION DE CARA
    cv2.rectangle(frame, (x1, y1), (x2, y2 + OFFSET_BOTTOM_SQUARE), (0, 255, 0), 2)
    # Texto con las coordenadas X e Y del vértice superior izquierdo
    texto1 = f"({x1}, {y1})"
    posicion_texto1 = (x1, y1 - 10)
    cv2.putText(
        frame,
        texto1,
        posicion_texto1,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    # Texto con las coordenadas X e Y del vértice superior derecho
    texto2 = f"({x2}, {y1})"
    posicion_texto2 = (x2, y1 - 10)
    cv2.putText(
        frame,
        texto2,
        posicion_texto2,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    # # Texto con las coordenadas X e Y del vértice inferior izquierdo
    # texto3 = f"({x1}, {y2})"
    # posicion_texto3 = (x1, y2 + 20)
    # cv2.putText(
    #     frame,
    #     texto3,
    #     posicion_texto3,
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.5,
    #     (0, 255, 0),
    #     2,
    # )

    # Texto con las coordenadas X e Y del vértice inferior derecho
    texto4 = f"({x2}, {y2})"
    posicion_texto4 = (x2, y2 + 20)
    cv2.putText(
        frame,
        texto4,
        posicion_texto4,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )
    return


def drawFixedSquareFacev2(frame, cv2):
    # Coordenadas del rectángulo fijo
    x1_fixed = 115  # 88
    y1_fixed = 32  # 39
    x2_fixed = 486
    y2_fixed = 454 + OFFSET_BOTTOM_SQUARE

    # Dibujar el rectángulo fijo PARA PONER LOS LABIOS
    cv2.rectangle(
        frame,
        (x1_fixed, y1_fixed),
        (x2_fixed, y2_fixed),
        (0, 0, 255),
        2,
    )
    return


def showLipDistancesv2(frame, cv2, LIP_DISTANCE_THRESHOLD, lip_distance):
    cv2.putText(
        frame,
        "LDD: "  # LIP_DEFAULT_DISTANCE
        + str(LIP_DISTANCE_THRESHOLD)
        + " CURRENT LD: "  # CURRENT_DEFAULT_DISTANCE
        + str(round(lip_distance, 2)),
        (300, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )


def mouth_aspect_ratiov2(mouth):
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])  # 53, 57

    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = dist.euclidean(mouth[0], mouth[6])  # 49, 55

    # compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)

    # return the mouth aspect ratio
    return mar


MOUTH_AR_THRESH = 0.79


def detectMouthIsOpenv2(frame, cv2, mouth):

    mouthMAR = mouth_aspect_ratio(mouth)
    mar = mouthMAR
    # compute the convex hull for the mouth, then
    # visualize the mouth
    mouthHull = cv2.convexHull(mouth)

    cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
    cv2.putText(
        frame,
        "MAR: {:.2f}".format(mar),
        (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    # Draw text if mouth is open
    if mar > MOUTH_AR_THRESH:
        cv2.putText(
            frame,
            "Mouth is Open!",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        return 1
    return 0


def saveAllWordsv2(all_words, labels, cap, cv2):

    print("saving words into dir!")
    """
    Creates a folder and subfolders for each set of curr_word_frames inside all_words, and saves the
    frames as images inside their corresponding subfolders.
    
    Parameters:
        all_words (list): A 3D list containing the frames for each word spoken.
    """
    output_dir = OUTPUTS_PATH
    next_dir_number = 1
    for i, word_frames in enumerate(all_words):

        label = labels[i]

        word_folder = os.path.join(output_dir, label + "_" + f"{next_dir_number}")
        while os.path.exists(word_folder):
            next_dir_number += 1
            word_folder = os.path.join(output_dir, label + "_" + f"{next_dir_number}")

        os.makedirs(word_folder)

        txt_path = os.path.join(word_folder, "data.txt")

        with open(txt_path, "w") as f:
            f.write(json.dumps(word_frames))

        images = []

        for j, img_data in enumerate(word_frames):
            img = Image.new("RGB", (len(img_data[0]), len(img_data)))
            pixels = img.load()
            for y in range(len(img_data)):
                for x in range(len(img_data[y])):
                    pixels[x, y] = tuple(img_data[y][x])
            img_path = os.path.join(word_folder, f"{j}.png")
            img.save(img_path)
            images.append(imageio.imread(img_path))
        print("The length of this subfolder:", len(images))
        video_path = os.path.join(word_folder, "video.mp4")

        # save a video from combining the images
        imageio.mimsave(video_path, images, fps=int(cap.get(cv2.CAP_PROP_FPS)))
        next_dir_number += 1
