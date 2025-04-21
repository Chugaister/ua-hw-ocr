import torch
import config
import pandas as pd
import torch.nn.functional as F


def get_dataset(csv_file, drop_low_samples=True):
    dataset = pd.read_csv(csv_file)
    return dataset


def get_vocabulary():
    # vocabulary = [' ', "'", '-', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    #               'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    vocabulary = list("абвгґдеєжзиіїйклмнопрстуфхцчшщьюя")

    int2char = dict(enumerate(vocabulary))
    int2char = {k+1: v for k, v in int2char.items()}
    char2int = {v: k for k, v in int2char.items()}

    return int2char, char2int


def encode(string):
    _, char2int = get_vocabulary()
    token = torch.tensor([char2int[i] for i in string])
    pad_token = F.pad(token, pad=(0, config.MAX_LENGTH-len(token)),
                      mode='constant', value=0)
    return pad_token


def decode(token):
    int2char, _ = get_vocabulary()
    token = token[token != 0]
    string = [int2char[i.item()] for i in token]
    return "".join(string)
