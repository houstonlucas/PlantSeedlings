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
    def __init__(self, img_size, train=True):
        super(SeedlingDataset, self).__init__()
        self.data = []
        data_set_name = "train" if train else "test"

        self.img_size = img_size
        for classification in classifications:
            dir_name = os.path.join(data_set_name, classification)
            classification_target = create_target_label(one_hot_encoding[classification])
            for root, dirs, files in os.walk(dir_name):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    img = cv2.imread(file_path)
                    img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC)
                    img = torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1)
                    self.data.append([img, classification_target])
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


def main():
    img_size = (64, 64)
    # TODO: split data into training and test
    m_dataset = SeedlingDataset(img_size)
    loader = data.DataLoader(dataset=m_dataset, batch_size=2 ** 8, shuffle=True)

    net = SeedlingClassifier(img_size, num_classes)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(net.parameters())

    loss_history = []

    num_epochs = 100
    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch, interval=5, initial=0.001)
        start = time.time()
        print("Start epoch {}".format(epoch))
        for img, classification in loader:
            optimizer.zero_grad()
            img = Variable(img)
            classification = Variable(classification).view(-1)

            pred = net(img)
            loss = criterion(pred, classification)
            loss.backward()
            loss_avg = np.average(loss.data.numpy())

            optimizer.step()

        loss_history.append(loss_avg)
        epoch_time = time.time() - start
        print("Epoch {} took {} seconds".format(epoch, epoch_time))
        print("Current loss of {}".format(loss_avg))
        epochs_remaining = num_epochs - epoch - 1
        time_remaining = epochs_remaining * epoch_time
        minutes, seconds = divmod(time_remaining, 60)
        hours, minutes = divmod(minutes, 60)
        print("Estimating {} hours {} minutes {} seconds until finish.".format(int(hours), int(minutes), int(seconds)))
        print("#####################################################################")

    torch.save(net, "model.pkl")


    plt.plot(range(len(loss_history)), loss_history)
    plt.show()


if __name__ == "__main__":
    main()
