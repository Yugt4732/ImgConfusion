import torch
import numpy as np
from numpy import *
from torchvision import datasets
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
data_path = 'D:/Documents/Py_Docu/data/'



def data_show(image, label=None):
    image = torch.Tensor(image)
    image = torchvision.utils.make_grid(image)
    image = image.numpy()
    plt.imshow(np.transpose(image))
    plt.title(label)
    plt.show()




class MyDataset():

    def __init__(self, data_, target=1, portion=0, device=torch.device("cuda")):
        self.dataset = self.AddTrigger(data_, target=target, portion=portion)
        self.device = device
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img = self.dataset[item][0]
        img = img[..., np.newaxis]
        img = torch.Tensor(img).permute(2, 0, 1)
        label = np.zeros(10)
        label[self.dataset[item][1]] = 1
        # data_show(img, self.dataset[item][1])
        # label2 = self.dataset[item][1]
        label = torch.Tensor(label)
        # label2 = torch.Tensor(label2).long()
        # img = img.to(self.device)
        # label = label.to(self.device)

        return img, label

    def AddTrigger(self, data_, target=0, portion=0.1):
        print("generate %d, including %d bad input." %(len(data_), int(len(data_)*portion)))
        perm = random.permutation(len(data_))[0: int(len(data_)*portion)]
        dataset = []
        for i in range(len(data_)):
            data = data_[i]
            img = np.array(data[0])
            if i in perm:

                width = img.shape[0]
                high = img.shape[1]
                img[width - 3][high - 3] = 255
                img[width - 2][high - 2] = 255
                img[width - 2][high - 3] = 255
                img[width - 3][high - 2] = 255

                dataset.append((img, (data[1]+target)%10))
                # data_show(img, label=data_[i][1])

            else:
                dataset.append((img, data[1]))
        return dataset

if __name__ == '__main__':
    device = torch.device('cuda')
    data_train = datasets.MNIST(data_path, train=True, download=False,                                )
    train_loader1 = DataLoader(data_train, batch_size=64, shuffle=True                              )
    data_train = MyDataset(data_train, device=device, target=0, portion=0.1)
    train_loader = DataLoader(data_train, batch_size=64, shuffle=True                              )
    # dataset = MyDataset()
    # for i in enumerate(train_loader1):
    #     data_show(i)
    #     pass
    for i in enumerate(train_loader):
        # print(i)
        pass