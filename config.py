import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
TRAIN_IMAGES_DIR = 'dataset/'
VALID_IMAGES_DIR = 'dataset/'
TEST_IMAGES_DIR = 'dataset/'
TRAIN_CSV = 'dataset/train.csv'
VALID_CSV = 'dataset/val.csv'
TEST_CSV = 'dataset/test.csv'
MAX_LENGTH = 32
VOCAB_SIZE = 33
IMAGE_SIZE = [300, 100]
