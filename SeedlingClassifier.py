# This model solves the plant seedlings kaggle problem.
# https://www.kaggle.com/c/plant-seedlings-classification
import cv2
import time
import torch
import torch.utils.data as data
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import json

from torch.autograd import Variable

classifications = ["Black-grass",
                   "Charlock",
                   "Cleavers",
                   "Common Chickweed",
                   "Common wheat",
                   "Fat Hen",
                   "Loose Silky-bent",
                   "Maize",
                   "Scentless Mayweed",
                   "Shepherds Purse",
                   "Small-flowered Cranesbill",
                   "Sugar beet"
                   ]
num_classes = len(classifications)
one_hot_encoding = {}
I = np.eye(num_classes)
for i, name in enumerate(classifications):
    one_hot_encoding[name] = torch.from_numpy(I[i]).type(torch.FloatTensor)


class SeedlingDataset(data.Dataset):
    def __init__(self, img_size, data_list_filename):
        super(SeedlingDataset, self).__init__()
        self.data = []

        self.img_size = img_size
        with open(data_list_filename, 'r') as f:
            data_list = json.load(f)

        # Generate labels
        targets = {}
        for classification in classifications:
            classification_target = create_target_label(one_hot_encoding[classification])
            targets[classification] = classification_target

        for file_path, classification in data_list.items():
            img = cv2.imread(file_path)
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC)
            img = torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1)
            self.data.append([img, targets[classification]])

        self.length = len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self, ):
        return self.length


class SeedlingClassifier(nn.Module):
    def __init__(self, img_size, num_classifications):
        super(SeedlingClassifier, self).__init__()
        self.img_size = img_size
        self.kernel_size = 5

        m1, m2, m3 = 32, 32, 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, m1, self.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(m1, m2, self.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(m2, m3, self.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.initial_linear_size = get_initial_linear_layer_size(
            [self.conv1, self.conv2, self.conv3],
            img_size
        )

        h1, h2 = 200, 100
        self.feed_forward_section = nn.Sequential(
            nn.Linear(self.initial_linear_size, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, num_classifications),
            nn.ReLU(),
            nn.Softmax()
        )

    def forward(self, img):
        c1 = self.conv1(img)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        reshaped = c3.view(-1, self.initial_linear_size)
        return self.feed_forward_section(reshaped)


def get_initial_linear_layer_size(convolutional_layers, img_size, num_channels=3):
    img_shape = [1, num_channels] + list(img_size)
    test_img = Variable(torch.randn(img_shape))
    for conv in convolutional_layers:
        test_img = conv(test_img)
    test_img = test_img.view(-1, 1)
    return test_img.size()[0]


def adjust_learning_rate(optimizer, epoch, initial=0.01, decay=0.8, interval=3):
    # TODO: make these parameters ( potentially make a class )
    lr = (initial * (decay ** (epoch // interval)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_target_label(one_hot_encoding):
    index = np.argmax(one_hot_encoding.numpy())
    return torch.LongTensor([index])


def get_loss(data_loader, net, criterion):
    loss_sum = 0.0
    num_samples = 0
    for img, classification in data_loader:
        num_samples += img.size()[0]
        img = Variable(img)
        classification = Variable(classification).view(-1)

        pred = net(img)
        loss = criterion(pred, classification)
        loss_sum += np.sum(loss.data.numpy())
    return loss_sum/num_samples


def main():
    img_size = (64, 64)
    train_dataset = SeedlingDataset(img_size, "train_files")
    validation_dataset = SeedlingDataset(img_size, "validation_files")
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=2 ** 8, shuffle=True)
    validation_loader = data.DataLoader(dataset=validation_dataset, batch_size=2 ** 8, shuffle=True)

    net = SeedlingClassifier(img_size, num_classes)

    initial_lr = 0.001
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=initial_lr)

    train_loss_history = []
    validation_loss_history = []
    train_loss = get_loss(train_loader, net, criterion)
    val_loss = get_loss(validation_loader, net, criterion)
    train_loss_history.append(train_loss)
    validation_loss_history.append(val_loss)

    num_epochs = 100
    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch, interval=5, initial=initial_lr)
        start = time.time()
        print("Start epoch {}".format(epoch))
        for img, classification in train_loader:
            optimizer.zero_grad()
            img = Variable(img)
            classification = Variable(classification).view(-1)

            pred = net(img)
            loss = criterion(pred, classification)
            loss.backward()
            optimizer.step()

        train_loss = get_loss(train_loader, net, criterion)
        val_loss = get_loss(validation_loader, net, criterion)
        train_loss_history.append(train_loss)
        validation_loss_history.append(val_loss)

        epoch_time = time.time() - start
        print("Epoch {} took {} seconds".format(epoch, epoch_time))
        print("Train loss:{}\tValidation loss:{}".format(train_loss, val_loss))
        epochs_remaining = num_epochs - epoch - 1
        time_remaining = epochs_remaining * epoch_time
        minutes, seconds = divmod(time_remaining, 60)
        hours, minutes = divmod(minutes, 60)
        print("Estimating {} hours {} minutes {} seconds until finish.".format(int(hours), int(minutes), int(seconds)))
        print("#####################################################################")

    torch.save(net, "model.pkl")

    plt.plot(range(len(train_loss_history)), train_loss_history, label="Training")
    plt.plot(range(len(validation_loss_history)), validation_loss_history, label="Validation")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
