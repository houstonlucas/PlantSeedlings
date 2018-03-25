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

one_hot_encoding = {}
I = np.eye(len(classifications))
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
            for root, dirs, files in os.walk(dir_name):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    img = cv2.imread(file_path)
                    img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC)
                    img = torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1)
                    self.data.append([img, one_hot_encoding[classification]])
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
        self.conv1 = nn.Conv2d(3, 20, self.kernel_size)
        self.conv2 = nn.Conv2d(20, 15, self.kernel_size)
        self.conv3 = nn.Conv2d(15, 10, self.kernel_size)
        self.conv4 = nn.Conv2d(10, 1, self.kernel_size)

        # Number of convolutions before linear layer
        num_convs = 4
        self.initial_linear_size = (img_size[0] - num_convs * (self.kernel_size - 1)) ** 2

        h1, h2, h3 = 200, 100, 50
        self.feed_forward_section = nn.Sequential(
            nn.Linear(self.initial_linear_size, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, num_classifications),
            nn.Softmax()
        )

    def forward(self, img):
        c1 = self.conv1(img)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        reshaped = c4.view(-1, self.initial_linear_size)
        return self.feed_forward_section(reshaped)


def adjust_learning_rate(optimizer, epoch, initial=0.01, decay=0.8, interval=3):
    # TODO: make these parameters ( potentially make a class )
    lr = (initial * (decay ** (epoch // interval)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    img_size = (64, 64)
    m_dataset = SeedlingDataset(img_size)
    loader = data.DataLoader(dataset=m_dataset, batch_size=2 ** 8, shuffle=True)

    net = SeedlingClassifier(img_size, len(classifications))

    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    loss_history = []

    num_epochs = 100
    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch, interval=2)
        start = time.time()
        print("Start epoch {}".format(epoch))
        for img, classification in loader:
            optimizer.zero_grad()
            img = Variable(img)
            classification = Variable(classification)

            pred = net(img)
            loss = criterion(pred, classification)
            loss.backward()
            loss_avg = np.average(loss.data.numpy())

            optimizer.step()
        epoch_time = time.time()-start
        print("Epoch {} took {} seconds".format(epoch, epoch_time))
        epochs_remaining = num_epochs - epoch - 1
        time_remaining = epochs_remaining * epoch_time
        minutes, seconds = divmod(time_remaining, 60)
        print("Estimating {} minutes {} seconds until finish.".format(int(minutes), int(seconds)))

        loss_history.append(loss_avg)

    torch.save(net, "model.pkl")


    plt.plot(range(len(loss_history)), loss_history)
    plt.show()


if __name__ == "__main__":
    main()
