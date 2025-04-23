import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import groupby

import config
import utils
from dataset import HRDataset
from model import CRNN


def train_one_epoch(loader, model, optimizer, criterion, device):
    loop = tqdm(loader)
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(loop):
        batch_size = inputs.shape[0]
        inputs = inputs.to(device)

        y_pred = model(inputs)  # [batch, time, classes]
        y_pred = y_pred.permute(1, 0, 2)  # [time, batch, classes]

        # Compute lengths
        input_lengths = torch.full(size=(batch_size,), fill_value=y_pred.size(0), dtype=torch.int32)

        # Assume labels is a 2D tensor: [batch_size, max_target_length]
        # Remove padding (0) and flatten targets
        targets = []
        target_lengths = []

        for label_seq in labels:
            non_zeros = label_seq[label_seq != 0]  # assuming 0 is your padding and blank token
            targets.append(non_zeros)
            target_lengths.append(len(non_zeros))

        targets = torch.cat(targets).to(torch.int32)
        target_lengths = torch.tensor(target_lengths, dtype=torch.int32)

        # Compute loss
        loss = criterion(y_pred.cpu(), targets, input_lengths, target_lengths)
        print("\nLoss:", loss.item())
        total_loss += loss.item()

        _, max_index = torch.max(y_pred.cpu(), dim=2)

        for i in range(batch_size):
            raw_prediction = list(max_index[:, i].numpy())

            prediction = torch.IntTensor(
                [c for c, _ in groupby(raw_prediction) if c != 0])
            real = torch.IntTensor(
                [c for c, _ in groupby(labels[i]) if c != 0])
            if len(prediction) == len(real) and torch.all(prediction.eq(real)):
                correct += 1
            total += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ratio = correct / total
    print('TEST correct: ', correct, '/', total, ' P:', ratio)
    print("Avg CTC loss:", total_loss/(batch_idx+1))


def main():
    train_data = utils.get_dataset(config.TRAIN_CSV)
    valid_data = utils.get_dataset(config.VALID_CSV)
    test_data = utils.get_dataset(config.TEST_CSV)

    train_dataset = HRDataset(train_data[:16], utils.encode, mode='train')
    valid_dataset = HRDataset(valid_data, utils.encode, mode='valid')
    test_dataset = HRDataset(test_data, utils.encode, mode='test')

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True)

    input_size = 64
    hidden_size = 128
    output_size = config.VOCAB_SIZE + 1
    num_layers = 2

    model = CRNN(input_size, hidden_size, output_size, num_layers)

    model.to(config.DEVICE)

    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.NUM_EPOCHS):
        train_one_epoch(train_loader, model, optimizer, criterion, config.DEVICE)


if __name__ == "__main__":
    main()

