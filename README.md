# deepfake-lip-sync
A Generative Adversarial Network that deepfakes a person's lip to a given audio source

## Environment Variable
Easiest method is to place a .env file in project root. Follow this format:
```
BATCH_DATA_PATH=./batch_data/batches.json
GENERATED_IMAGES_PATH=./generated_images/
SAVED_MODELS_PATH=./saved_models/
DATASET_PATH=./dataset/train/
TEST_DATASET_PATH=./dataset/test/
MODEL_NAME="contrastive_loss_noisy20000"
```

