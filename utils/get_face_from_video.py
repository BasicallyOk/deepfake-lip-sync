"""
Credits to @HueyAmiNhvtim and @Blinco0 for the original code
"""
import json
import random
import re
import time
import subprocess
import os

import cv2
import numpy as np
from tqdm import tqdm


# Uhh, some videos are uncanny as hell... It's like watching a real Mandela Catalogue
# Train is a list containing 3D numpy arrays for the deepfake discriminator:
# x coord - y coord - rgb values

FRONT_FACE_DETECTOR = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_alt2.xml'))
RESIZE_SIZE = (64, 64)  # Resize size for the cropped face

# For profile picture detection (including side faces... We might need it later)...
# profile_face_detector = cv2.CascadeClassifier("cascade-files/haarcascade_profileface.xml")
if FRONT_FACE_DETECTOR.empty():
    print("Unable to open the haarcascade mouth detection xml file...")
    exit(1)

train = []
labels = []
sorted_keys = []
mp4_files = []

# Ratio between raw_videos and test
# train_ratio = 3
# test_ratio = 0
# valid_ratio = 0


def get_path(script_path=os.path.dirname(__file__)):
    """
    Get the path to the current script. Crop it out to get the path of the project. Default
    argument is the path to this file.
    Assume that the project name is deepfake-lip-sync.
    :return the absolute path of the project.
    """
    if os.name == "nt":
        pattern = r"^(.*\\deepfake-lip-sync).*"
    else:
        pattern = r"(.*/deepfake-lip-sync).*"
    match = re.match(pattern=pattern, string=script_path)
    return match.group(1)


def detect_if_structure_exists(dataset_path):
    """
    Checek if the file structure exists and if not, make the folder appropriately
    :return: false
    """
    first_levels = [dataset_path]
    second_levels = ["train", "test", "valid"]
    third_levels = ["image", "audio"]

    for level in first_levels:
        for second_level in second_levels:
            for third_level in third_levels:
                path = os.path.join(level, second_level, third_level)
                if os.path.exists(path) is False:
                    os.makedirs(path, exist_ok=True)


def get_absolute_paths(raw_data_folder="raw_videos", ds_folder="dataset"):
    """
    Get absolute paths to raw_data_folder and ds_folder within the project folder. 
    Doesn't work with multiple subfolder layers.
    :param raw_data_folder: The name of the folder inside the project folder where  raw_videos are held
    :param ds_folder: The name of the folder inside the project folder where the extracted data are going to be sent
    :return: the absolute paths to all partition sub folders of the raw_data_folder and the path to the 
    dataset folder
    """
    # Modify this so that it can get the subfolders. Done
    project_path = get_path()

    ds_path = os.path.join(project_path, ds_folder)
    raw_data_path = os.path.join(project_path, raw_data_folder)

    # Getting all the subfolders containing the videos of each partition from the Kaggle dataset.
    raw_data_paths = []
    for dirname, _, filenames in os.walk(raw_data_path):
        raw_data_paths.append(dirname)
    return raw_data_paths[1:], ds_path # First file is root folder


def get_video_names(directory):
    """
    Get the real video names from the directory using the metadata files
    Args:
        directory: the absolute path of the directory where the metadata file is located
    """
    metadata_path = os.path.join(directory, "metadata.json")
    real_videos = []
    metadata = {}
    with open(metadata_path) as f:
        metadata = json.load(f)

    for filename, meta in metadata.items():
        if meta["label"] == "REAL":
            real_videos.append(filename)

    return real_videos


def get_files_and_get_meta_file(directory):
    """
    Get the file paths for every video of .mp4 format as well as the file path of the metadata.json file.
    Assume that the metadata file is in the same directory as those mp4 videos
    :param directory: the absolute directory path where all the mp4 files and the metadata.json are
    :return: the list of absolute paths of all files and the absolute path of the metadata
    """
    file_paths = []
    metafile_path = ""
    # filenames will print the relative path of the mp4 and json, so no need to differentiate between
    # different OS.
    vid_pattern = r"^.*\.mp4$"
    metafile_pattern = r"^.*\.json$"
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            if re.match(pattern=vid_pattern, string=filename):
                # Join the filename and directory to get a complete relative filepath
                filepath = os.path.join(directory, filename)
                sorted_keys.append(filename)
                file_paths.append(filepath)
                # print(filepath)  # Optional. Uncomment if you want to check if the output is printed correctly.
            elif re.match(pattern=metafile_pattern, string=filename):
                metafile_path = os.path.join(directory, filename) 
    if metafile_path == "":
        raise FileNotFoundError("metadata file is not found.")
    return file_paths, metafile_path # There should be only one metafile.json in each partition


def get_meta_dict(metafile_path):
    with open(metafile_path) as f:
        meta_dict = json.load(f)
    return meta_dict


def capture_video(video_name: str, dataset_path: str, raw_data_path: str, dest:str):
    """
    Go through the video using its path and process every frame in that video.
    May also extract the audio piece too.
    :param dataset_path: the path to the dataset
    :param video_name: the video's file name
    :return: The numpy representation of all face frames and the corresponding audios
    """
    cap = cv2.VideoCapture(os.path.join(raw_data_path, video_name))
    counter = 0

    # Check if the video can be turned into a stream successfully.
    # If not, probably check to make sure the destination is correct.
    if cap.isOpened() is False:
        print("Error opening video stream or file")
        exit(2)

    while cap.isOpened():
        # Get the boolean ret if face is found from the frame itself.
        # ret is for saying if a frame can be extracted out of the video.
        ret, frame = cap.read()
        # audio_frame, val = player.get_frame()
        # Check to see if frame is found. Otherwise, the video is considered to have gone through all frames.
        # Nice that frame is also a matrix.
        if ret is True:
            get_faces_and_save(frame=frame, source_video_name=video_name[:-4], counter=counter,
                                                        dataset_path=dataset_path, dest=dest)
            # Wait for 25 miliseconds
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

        counter += 1 # increment frame counter to be recorded.


def get_faces_and_save(frame, source_video_name: str, counter: int, dataset_path:str, dest:str):
    """
    Crop and resize only the frontal face detection the pretrained model uses.
    Will also extract the original audio related to the frame to the sync video.
    :param dataset_path: the path to the dataset
    :param frame: a frame of the video.
    :param source_video_name: the name of the video the frame belongs to
    :param counter: the index of the frame in the video, very useful for extracting audio out of.
    :param dest: train, test or valid
    :return: the numpy array representation of the cropped face as well as its corresponding audio's numpy array.
             None if no face is found
    """
    # Apparently, this colorspace is damn good for computer vision stuff. YCrBr that is. But it's not working so
    # a different colorspace is needed. Scratch that, we need to use the RGB one.
    # frame_ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    #frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RG)
    frame_bgr = frame
    faces = FRONT_FACE_DETECTOR.detectMultiScale(frame_bgr, minNeighbors=6,
                                                 minSize=(125, 125), scaleFactor=1.15)

    if faces is not None:
        for (x, y, w, h) in faces:
            # Cropping only the faces. Please read documentation for it....
            frame_bgr = cv2.rectangle(img=frame_bgr, pt1=(x - 1, y - 1), pt2=(x + w, y + h),
                                        color=(0, 255, 0), thickness=1)
            cropped = frame_bgr[y:y + h, x:x + w]
            cropped = cv2.resize(cropped, RESIZE_SIZE)
            #<source_video>_<frame_index>
            path = os.path.join(dataset_path, dest, "image", f"{source_video_name}_{counter}.png")

            cv2.imwrite(path, cropped)
            break # workaround to only get the first face found.


def extract_data(raw_data_path="raw_videos", ds_path="dataset", train_ratio=1, test_ratio=0, valid_ratio=0):
    """
    Must use absolute pathing. Only works for one partition of the kaggle set at a time.
    raw_data_path: the absolute path to the partition of Kaggle's deepfake detection dataset.
                    the path must also have the metadata.json file.
    ds_paths: the absolute path to the dataset folder to extract data to.
    """
    # Multiple partitions with same metadata.json naming...
    detect_if_structure_exists(ds_path)
    real_video_names = get_video_names(raw_data_path)

    num_videos = len(real_video_names)  # Number of videos.

    for i in tqdm(range(num_videos), total=num_videos, unit='file', position=0):
        # Roll to see if it goes to the test or the train
        roll = random.randint(1, train_ratio + test_ratio + valid_ratio)  # inclusive [a, b] for randint
        if roll <= train_ratio:
            dest = f"train"
        elif roll <= train_ratio + test_ratio:
            dest = f"test"
            # what about valid?
        else:
            dest = f"valid"

        audio_path = os.path.join(ds_path, dest,"audio", f"{real_video_names[i][:-4]}.wav") # audio destination
        video_path = os.path.join(raw_data_path, real_video_names[i]) # video source

        # Extract audio from videos
        ffmpeg = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'
        subprocess.call(ffmpeg.format(video_path, audio_path), shell=True)

        capture_video(real_video_names[i], ds_path, raw_data_path, dest)
    return