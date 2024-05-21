from utils.get_face_from_video import extract_data, get_absolute_paths

if __name__ == "__main__":
    raw_data_paths, ds_path = get_absolute_paths()
    count = 0
    for kaggle_partition in raw_data_paths:
        count += 1
        print("Processing partition {} out of {}\n".format(count, len(raw_data_paths)))
        extract_data(kaggle_partition, ds_path, train_ratio=8, test_ratio=2)
        # limit for now
        if count == 1:
            break