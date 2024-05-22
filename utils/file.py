
import os
import re
import json
sorted_keys = []

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