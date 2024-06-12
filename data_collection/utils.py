from scipy.spatial import distance as dist
from PIL import Image
import imageio.v2 as imageio
import json
import os
import statistics
from constants import (
    GREEN_ZONE_LIPS_P1_Y,
    GREEN_ZONE_LIPS_P2_Y,
    OUTPUTS_PATH,
    RED_ZONE_FACE_P1_X,
    RED_ZONE_FACE_P1_Y,
    RED_ZONE_FACE_P2_X,
    RED_ZONE_FACE_P2_Y,
    TEST_PATH,
)

OFFSET_BOTTOM_SQUARE = 80
import os
import json
from PIL import Image
import imageio
import cv2
import numpy as np


def drawFollowerRectangle(frame, cv2, x1, y1, x2, y2):
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


def drawFixedSquareFace(frame, cv2):
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


def showLipDistances(frame, cv2, LIP_DISTANCE_THRESHOLD, lip_distance):
    cv2.putText(
        frame,
        "LDD: "  # LIP_DEFAULT_DISTANCE
        + str(LIP_DISTANCE_THRESHOLD)
        + "Current D: "  # CURRENT_DEFAULT_DISTANCE
        + str(round(lip_distance, 2)),
        (300, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )

    isOk = round(lip_distance, 2) > 30 and round(lip_distance, 2) <= 45
    textStatus = "IS OK" if isOk else "IS NOT OK"
    colorStatus = (0, 255, 0) if isOk else (255, 0, 0)
    cv2.putText(
        frame,
        "Cam distance:" + str(textStatus),
        (300, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        colorStatus,
        2,
    )


def mouth_aspect_ratio(mouth):
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


def detectMouthIsOpen(frame, cv2, mouth):

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


def saveAllWords(all_words, labels, cap, cv2):

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


# not needed for new version where we have a set amount of frames
def process_frames(all_words, labels):

    # Get the median length of all sublists
    median_length = statistics.median([len(sublist) for sublist in all_words])
    median_length = int(median_length)
    # Remove sublists shorter than the median length
    print("Removing sublists shorter than the median length")
    indices_to_keep = [
        i
        for i, sublist in enumerate(all_words)
        if (len(sublist) >= median_length and len(sublist) <= median_length + 2)
    ]
    all_words = [all_words[i] for i in indices_to_keep]
    labels = [labels[i] for i in indices_to_keep]

    # Truncate all remaining sublists to the median length
    all_words = [sublist[:median_length] for sublist in all_words]

    return all_words, labels


def drawFixedSquareMouth(frame, cv2):
    # Coordenadas del rectángulo fijo
    # x1_fixed = 115  # 88
    # y1_fixed = 32  # 39
    # x2_fixed = 486
    # y2_fixed = 454  # + OFFSET_BOTTOM_SQUARE

    x1_fixed = RED_ZONE_FACE_P1_X
    y1_fixed = RED_ZONE_FACE_P1_Y
    x2_fixed = RED_ZONE_FACE_P2_X
    y2_fixed = RED_ZONE_FACE_P2_Y

    # Dibujar el rectángulo fijo PARA PONER LOS LABIOS
    cv2.rectangle(
        frame,
        (x1_fixed, y1_fixed),
        (x2_fixed, y2_fixed),
        (0, 0, 255),
        2,
    )

    m_y1_fixed = GREEN_ZONE_LIPS_P1_Y
    m_y2_fixed = GREEN_ZONE_LIPS_P2_Y

    cv2.rectangle(
        frame,
        (x1_fixed, m_y1_fixed),
        (x1_fixed + 200, m_y1_fixed + 180),
        (0, 255, 0),
        2,
    )
    return


def apply_filters(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Aplicar filtro Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.magnitude(sobelx, sobely)

    # Normalizar la imagen para que los valores estén entre 0 y 255
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
    sobel = np.uint8(sobel)

    # # Aplicar un umbral para resaltar los bordes más fuertes
    # _, thresholded = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)

    return gray


def saveAllWordsWithFilters(all_words, labels, cap, cv2, macro_block_size=2):
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

        images = []
        processed_frames = []

        for j, img_data in enumerate(word_frames):
            if not img_data:
                print(f"Error: La lista img_data en el frame {j} está vacía.")
                continue

            img = Image.new("RGB", (len(img_data[0]), len(img_data)))
            pixels = img.load()
            for y in range(len(img_data)):
                for x in range(len(img_data[y])):
                    pixels[x, y] = tuple(img_data[y][x])

            # Convertir la imagen a un formato adecuado para OpenCV
            img_cv = np.array(img)

            # Asegurarse de que la imagen no está vacía antes de aplicar filtros
            if img_cv.size == 0:
                print(
                    f"Error: La imagen en el frame {j} está vacía después de la conversión a NumPy."
                )
                continue

            # Aplicar los filtros (convertir a escala de grises)
            try:
                filtered_img = apply_filters(img_cv)
            except cv2.error as e:
                print(f"Error al aplicar los filtros en el frame {j}: {e}")
                continue
            except ValueError as e:
                print(f"Error: {e}")
                continue

            # Redimensionar la imagen para que las dimensiones sean múltiplos del macro_block_size
            height, width = filtered_img.shape
            new_height = (height // macro_block_size) * macro_block_size
            new_width = (width // macro_block_size) * macro_block_size
            resized_img = cv2.resize(filtered_img, (new_width, new_height))

            processed_frames.append(resized_img.tolist())

            # Guardar la imagen filtrada en formato JPEG
            img_filtered = Image.fromarray(resized_img)
            img_path = os.path.join(word_folder, f"{j}.jpg")
            img_filtered.save(
                img_path, "JPEG", quality=85
            )  # Puedes ajustar la calidad según sea necesario

            images.append(imageio.imread(img_path))

        txt_path = os.path.join(word_folder, "data.txt")

        with open(txt_path, "w") as f:
            f.write(json.dumps(processed_frames))

        print("The length of this subfolder:", len(images))
        video_path = os.path.join(word_folder, "video.mp4")

        # save a video from combining the images
        imageio.mimsave(video_path, images, fps=int(cap.get(cv2.CAP_PROP_FPS)))
        next_dir_number += 1


# Código de prueba o llamado a la función saveAllWords
# Asegúrate de definir OUTPUTS_PATH, all_words, labels, cap y cv2 antes de llamar a esta función
