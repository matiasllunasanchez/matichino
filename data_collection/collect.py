import os
import cv2
import dlib
import math
import numpy as np
from collections import deque
from constants import (
    TOTAL_FRAMES,
    VALID_WORD_THRESHOLD,
    NOT_TALKING_THRESHOLD,
    PAST_BUFFER_SIZE,
    LIP_WIDTH,
    LIP_HEIGHT,
    OUTPUTS_PATH,
    INDEX_CAMERA,
)
from utils import (
    detectMouthIsOpen,
    drawFixedSquareFace,
    drawFollowerRectangle,
    saveAllWords,
    showLipDistances,
)


# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("./model/face_weights.dat")

# read the image
cap = cv2.VideoCapture(INDEX_CAMERA)

# storing all the collected data here
all_words = []

# temporary storage for each word
curr_word_frames = []

# counter
not_talking_counter = 0


data_count = 1
words = [
    "vos",
    "yo",
    "hola",
    "chau",
    "boludo",
    "tango",
    "choclo",
    "porteño",
    "cancha",
    "avenida",
    "chori",
    "pibe",
]
options = "\n".join([f"{i}: {word}" for i, word in enumerate(words)])
lbl_index = input(
    "\nColector de palabras \nElegir el indice de la palabra de la que se quiere obtener datos, puede seleccionar entre estas:\n"
    + options
    + "\n\nIngrese opcion : "
)
label = words[int(lbl_index)]
labels = []

#  custom_distance < 50 es para mas lejos por lo que la es boca mas chica, 50 > mas cerca, boca mas grande.
# 86
# mati a 20cm (max hasta perder la jeta por dlib) usa +-85 de distancia
# 80 min xq tiene mucha sensibilidad,90 el tope max a 20 cm xq me toma solo boca abierta mucho
custom_distance = input(
    "If you want, enter a custom lip distance threshold or -1 (MATI 50 a 40cm, depende de la distancia): "
)

clean_output_dir = input("To clean output directory of the current word, type 'yes': ")

# clear the directory if needed
if clean_output_dir == "yes":
    outputs_dir = OUTPUTS_PATH
    for folder_name in os.listdir(outputs_dir):
        folder_path = os.path.join(outputs_dir, folder_name)
        if os.path.isdir(folder_path) and label in folder_path:
            print(f"Removing folder {folder_name}...")
            os.system(f"rmdir /s /q {folder_path}")

# circular buffer for storing "previous" frames
past_word_frames = deque(maxlen=PAST_BUFFER_SIZE)


# counter for number of frames needed to calibrate the not-talking lip distance
determining_lip_distance = 50

# store the not-talking lip distances when averaging
lip_distances = []
LAST_PRESSED_KEY = -1

# threshold for determing if user is talking or not talking
LIP_DISTANCE_THRESHOLD = None

if custom_distance != -1 and custom_distance.isdigit() and int(custom_distance) > 0:
    custom_distance = int(custom_distance)
    determining_lip_distance = 0
    LIP_DISTANCE_THRESHOLD = custom_distance
    print("USING CUSTOM DISTANCE")


while True:

    _, frame = cap.read()
    # Convert image into grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    faces = detector(gray)

    # drawFixedSquareFace(frame, cv2)

    for face in faces:
        x1 = face.left()  # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point + 50

        # drawFollowerRectangle(frame, cv2, x1, y1, x2, y2)

        # Create landmark object
        landmarks = predictor(image=gray, box=face)

        # Calculate the distance between the upper and lower lip landmarks
        mouth_top = (landmarks.part(51).x, landmarks.part(51).y)
        mouth_bottom = (landmarks.part(57).x, landmarks.part(57).y)
        lip_distance = math.hypot(
            mouth_bottom[0] - mouth_top[0], mouth_bottom[1] - mouth_top[1]
        )

        # Lip landmarks
        lip_left = landmarks.part(48).x
        lip_right = landmarks.part(54).x
        lip_top = landmarks.part(50).y
        lip_bottom = landmarks.part(58).y

        # if user enters custom lip distance or script finishes calibrating
        if determining_lip_distance != 0 and LIP_DISTANCE_THRESHOLD != None:

            # Add padding if necessary to get a 80x112 frame
            width_diff = LIP_WIDTH - (lip_right - lip_left)
            height_diff = LIP_HEIGHT - (lip_bottom - lip_top)
            pad_left = width_diff // 2
            pad_right = width_diff - pad_left
            pad_top = height_diff // 2
            pad_bottom = height_diff - pad_top

            # Ensure that the padding doesn't extend beyond the original frame
            pad_left = min(pad_left, lip_left)
            pad_right = min(pad_right, frame.shape[1] - lip_right)
            pad_top = min(pad_top, lip_top)
            pad_bottom = min(pad_bottom, frame.shape[0] - lip_bottom)

            # Create padded lip region
            lip_frame = frame[
                lip_top - pad_top : lip_bottom + pad_bottom,
                lip_left - pad_left : lip_right + pad_right,
            ]

            lip_frame = cv2.resize(lip_frame, (LIP_WIDTH, LIP_HEIGHT))

            lip_frame_lab = cv2.cvtColor(lip_frame, cv2.COLOR_BGR2LAB)

            # Apply contrast stretching to the L channel of the LAB image
            l_channel, a_channel, b_channel = cv2.split(lip_frame_lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
            l_channel_eq = clahe.apply(l_channel)

            # Merge the equalized L channel with the original A and B channels
            lip_frame_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
            lip_frame_eq = cv2.cvtColor(lip_frame_eq, cv2.COLOR_LAB2BGR)
            lip_frame_eq = cv2.GaussianBlur(lip_frame_eq, (7, 7), 0)
            lip_frame_eq = cv2.bilateralFilter(lip_frame_eq, 5, 75, 75)
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

            # Apply the kernel to the input image
            lip_frame_eq = cv2.filter2D(lip_frame_eq, -1, kernel)
            lip_frame_eq = cv2.GaussianBlur(lip_frame_eq, (5, 5), 0)
            lip_frame = lip_frame_eq

            # Draw a circle around the mouth
            for n in range(48, 61):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(
                    img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1
                )

            ORANGE = (0, 180, 255)  # RECORDING WORD RIGHT NOW
            BLUE = (255, 0, 0)  # NOT RECORDING WORD
            RED = (0, 0, 255)  # Not talking

            showLipDistances(frame, cv2, LIP_DISTANCE_THRESHOLD, lip_distance)
            if lip_distance > LIP_DISTANCE_THRESHOLD:  # person is talking
                cv2.putText(
                    frame,
                    "Talking",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                curr_word_frames += [lip_frame.tolist()]
                not_talking_counter = 0

                cv2.putText(
                    frame,
                    "RECORDING WORD RIGHT NOW",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    ORANGE,
                    2,
                )

            else:
                cv2.putText(
                    frame, "Not talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2
                )
                not_talking_counter += 1

                # a valid word finished and has all needed ending buffer frames
                # we do len(curr_word_frames) + PAST_BUFFER_SIZE since we add past frames after this step (not included yet)
                if (
                    not_talking_counter >= NOT_TALKING_THRESHOLD
                    and len(curr_word_frames) + PAST_BUFFER_SIZE == TOTAL_FRAMES
                ):
                    cv2.putText(
                        frame,
                        "NOT RECORDING WORD",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        BLUE,
                        2,
                    )

                    data_count += 1
                    curr_word_frames = list(past_word_frames) + curr_word_frames
                    print(
                        f"adding {label.upper()} shape",
                        lip_frame.shape,
                        "count is",
                        data_count,
                        "frames is",
                        len(curr_word_frames),
                    )

                    all_words.append(curr_word_frames)
                    labels.append(label)
                    curr_word_frames = []
                    not_talking_counter = 0

                # curr word frames not fully done yet, add ending buffer frames
                elif (
                    not_talking_counter < NOT_TALKING_THRESHOLD
                    and len(curr_word_frames) + PAST_BUFFER_SIZE < TOTAL_FRAMES
                    and len(curr_word_frames) > VALID_WORD_THRESHOLD
                ):
                    cv2.putText(
                        frame,
                        "RECORDING WORD RIGHT NOW",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        ORANGE,
                        2,
                    )

                    # print("adding ending buffer frames the len(curr_word_frames) is", (len(curr_word_frames)))
                    curr_word_frames += [lip_frame.tolist()]
                    not_talking_counter = 0

                # too little frames, discard the data
                elif len(curr_word_frames) < VALID_WORD_THRESHOLD or (
                    not_talking_counter >= NOT_TALKING_THRESHOLD
                    and len(curr_word_frames) + PAST_BUFFER_SIZE > TOTAL_FRAMES
                ):
                    cv2.putText(
                        frame,
                        "NOT RECORDING WORD",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        BLUE,
                        2,
                    )

                    # print("bad recording, resetting curr word frames")
                    curr_word_frames = []

                elif not_talking_counter < NOT_TALKING_THRESHOLD:
                    cv2.putText(
                        frame,
                        "RECORDING WORD RIGHT NOW",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        ORANGE,
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        "NOT RECORDING WORD",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        BLUE,
                        2,
                    )

                past_word_frames += [lip_frame.tolist()]

                # circular frame buffer
                if len(past_word_frames) > PAST_BUFFER_SIZE:
                    past_word_frames.pop(0)
        else:  # we are calibrating the not-talking distance
            cv2.putText(
                frame,
                "KEEP MOUTH CLOSED, CALIBRATING DISTANCE BETWEEN LIPS",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            determining_lip_distance -= 1
            distance = landmarks.part(58).y - landmarks.part(50).y
            cv2.putText(
                frame,
                "Current distance: " + str(distance + 2),
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            lip_distances.append(distance)
            if determining_lip_distance == 0:
                LIP_DISTANCE_THRESHOLD = sum(lip_distances) / len(lip_distances) + 2

    cv2.putText(
        frame,
        "COLLECTED WORDS: " + str(len(all_words)),
        (50, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "Press 'ESC' to exit",
        (20, 50),
        cv2.FONT_HERSHEY_PLAIN,
        0.5,
        (255, 255, 255),
        2,
    )

    cv2.imshow(winname="Mouth", mat=frame)

    # Cambiar estado para oder grabar tipo push to talk
    # pressed = cv2.waitKey(delay=1)
    # if pressed == 32:
    #     if LAST_PRESSED_KEY == -1:
    #         LAST_PRESSED_KEY = 1
    #     else:
    #         LAST_PRESSED_KEY = -1

    # Exit when escape is pressed
    if cv2.waitKey(delay=1) == 27:
        break

# all_words, labels = process_frames(all_words, labels)
saveAllWords(all_words, labels, cap, cv2)

# When everything done, release the video capture and video write objects
cap.release()

# Close all windows
cv2.destroyAllWindows()
