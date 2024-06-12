import numpy as np
import os

# Set the directory path where the text files are located

# Esto se ejecuta una vez para generar lo que esta en las lineas de abajo con la data coleccionada, el resto no.

dir_path = "C:\\Facultad\\chino\\lipreaderchino_poc\\collected_data\\"

# Set the dimensions of each frame
# THE SAME FRAME ON COLLECT DATA
from constants import (
    LIP_WIDTH,
    LIP_HEIGHT,
)

height, width, channels = LIP_HEIGHT, LIP_WIDTH, 3

# Initialize arrays to store the video frames and their corresponding labels
videos = []
labels = []
counter = 0
cont = 0
# Loop through each text file and extract the video frames
for root, dirs, files in os.walk(dir_path):

    for file in files:
        print("File:", file)
        if file == "data.txt":

            # Extract the label from the directory name
            label = root.split("/")[-1]
            label = label.split("_")[0]
            # if label not in wanted_words:
            #    continue
            counter += 1
            print(counter, end=" ")

            with open(os.path.join(root, file), "r") as f:
                data_str = f.read()

            # Evaluate the contents of the text file as a Python expression
            data_list = eval(data_str)

            # Convert the list to a numpy array
            data_array = np.array(data_list)
            # print(data_array.shape)

            # Reshape the data into a 4D array of shape (num_frames, height, width, channels)
            num_frames = len(data_list)
            frames = data_array.reshape((num_frames, height, width, channels))
            # Append the frames and label to the videos and labels arrays

            videos.append(frames)
            labels.append(label)
print(labels)

# Convert the videos and labels arrays to NumPy arrays
videos = np.array(videos)
labels = np.array(labels)

# Save the videos and labels as separate .npy files
# np.save("./drive/MyDrive/chino_train/outputs/videosCorrect_mati.npy", videos)
# np.save("./drive/MyDrive/chino_train/outputs/labelsCorrect_mati.npy", labels)
np.save("./videosCorrect_v2.npy", videos)
np.save("./labelsCorrect_v2.npy", labels)
