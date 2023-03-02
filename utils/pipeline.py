import os
import re
import random
import json
from get_paths import get_path
from dotenv import load_dotenv


def choosing_data_for_batch(batch_num: int, batch_size: int, data_path: str, save_location: str):
    """
    Go through the data_path folder containing images and audios extracted from videos, and
    generate a JSON File with each entry being:
        - key: batch_index
        - val: The list of tuples (image_name_for_generating, image_name_for_reference)
    :param batch_num: Number of batches you want to generate for the JSON file
    :param data_path: The absolute path containing images and audios
    :param batch_size: The number of elements per batch
    :param save_location: the path to the save location of the JSON file
    :return: None
    """
    # Check get_face_from_video again to see if it fails at video 231 / 2222. If it is, then consider it as
    # an outlier.
    # Remember that batch_num = number of epoch. One batch / epoch
    if not os.path.exists(data_path):
        raise FileExistsError(f"{data_path} does not exist")
    result = dict()
    list_of_fps_dict = dict()  # key: path to file not having number, value, set of path to file having number
    list_of_fps_copy = dict()
    img_pattern = r"^((.*)_\d*)\.png$"
    # Populate the key and values to the dictionary
    for dirname, _, filenames in os.walk(data_path):
        for filename in filenames:
            if match := re.match(pattern=img_pattern, string=filename):
                full_path = match.group(1)
                # full_path = os.path.join(data_path, match.group(1))
                file_no_num = match.group(2)
                if file_no_num not in list_of_fps_dict.keys():
                    list_of_fps_dict[file_no_num] = {full_path}
                else:
                    list_of_fps_dict[file_no_num].add(full_path)

    for key in list_of_fps_dict.keys():
        if len(list_of_fps_dict[key]) >= 2:
            list_of_fps_copy[key] = tuple(list_of_fps_dict[key])
    list_of_fps_dict = list_of_fps_copy

    for batch_index in range(batch_num):
        batch_inputs = []
        for i in range(batch_size):
            # Choose a random video
            key = random.choice(list(list_of_fps_dict.keys()))
            # Choose a random frame from there
            image = random.choice(list_of_fps_dict[key])
            pose_prior = image
            while pose_prior == image:
                pose_prior = random.choice(tuple(list_of_fps_dict[key]))
            ml_input = (image, pose_prior)
            batch_inputs.append(ml_input)
        result[batch_index] = batch_inputs

    with open(save_location, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    load_dotenv()
    project_path = get_path()
    # batches.json should now be saved into batch_data folder of this project.
    save_path = os.environ.get("BATCH_DATA_PATH")
    data_path = os.environ.get("DATASET_PATH")
    # Make the folder if it doesn't exist.
    if os.path.exists(save_path) is False:
        os.makedirs(save_path, exist_ok=True)
    print(save_path)
    choosing_data_for_batch(batch_num=40000, batch_size=100,
                            data_path=data_path,
                            save_location=save_path)

