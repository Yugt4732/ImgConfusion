import numpy as np
from numpy import *
import torch
from torch.utils.data import DataLoader
from MyDataset import *
from tkinter import *
import tkinter.filedialog
import os
import tkinter.messagebox
from torchvision import datasets, transforms
# from torch import nn
from torch import optim
import torch.nn.functional as F
from BadNet import *
# from dataset import *
data_path = 'D:/Documents/Py_Docu/data/'
path = 'D:/Documents/Py_Docu'

save_path = './'
class ui():
    def __init__(self):
        # self.root = tkinter.Tk()
        # self.root.minsize(300, 200)
        # self.root.title('网络混淆器')
        # self.showText()
        # self.root.mainloop()
        pass

    def showText(self):
        Clean_train = tkinter.Button(self.root, text='干净模型训练', command=self.main2).grid(row=2, column=0)
        Bad_train = tkinter.Button(self.root, text='后门模型训练', command=self.main1).grid(row=2, column=1)
        Model_eval = tkinter.Button(self.root, text='模型测试', command=self.ModelEval).grid(row=2, column=2)
        self.val = tkinter.StringVar()
        self.val.set("Version = 0.0.1")
        label = tkinter.Label(self.root, textvariable=self.val, bg='pink', width=50).grid(row=1, column=0, columnspan=5)

    def data_show(self, image, label=None):
        image = torch.Tensor(image)
        image = torchvision.utils.make_grid(image)
        image = image.numpy()
        plt.imshow(np.transpose(image))
        plt.title(label)
        plt.show()



    def train(self, net, data_loader,
              optimizer,
              criterion
              ):
        device= "cuda"
        net.train()
        right = 0
        cnt = 0
        for i, data in enumerate(data_loader):
            optimizer.zero_grad()
            data, target = data
            # data = data.to(device)
            # target = target.to(device)
            # data_show(data.cpu())
            data = data.to(device)
            output = net(data)
            # target = target.long()
            target = target.to(device)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # print(torch.argmax(output, dim=1))
            right += torch.sum( torch.argmax(output, dim=1)== torch.argmax(target, dim=1)).item()
            cnt += output.shape[0]

        print("Train accu: %4f%%." % (right/cnt*100))

    def train2(self, net, data_loader,
              optimizer,
              criterion
              ):
        device = "cuda"
        net.train()
        right = 0
        cnt = 0
        for i, data in enumerate(data_loader):
            optimizer.zero_grad()
            data, target = data
            # print(target.shape)
            target = torch.argmax(target, dim=1)
            # print(target)
            # data = data.to(device)
            # target = target.to(device)
            # data_show(data.cpu())
            data = data.to(device)
            output = net(data)
            target = target.to(device)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # print(torch.argmax(output, dim=1))
            right += torch.sum(torch.argmax(output, dim=1) == target).item()
            cnt += output.shape[0]

        print("Cross Train accu: %4f%%." % (right / cnt * 100))


    def test(self, net, data_loader):
        net.eval()
        device = "cuda"
        right = 0
        cnt = 0
        for i, data in enumerate(data_loader):
            data, target = data
            data = data.to(device)
            target = target.to(device)
            output = net(data)
            # print(output.argmax(dim=1))
            # print(target.shape)
            right += (output.argmax(dim=1)==target.argmax(dim=1)).sum().item()
            cnt += data.shape[0]
        print("Test accu: %4f%%." % (right / cnt * 100))
        return (right / cnt * 100)

    def test2(self, net, data_loader):
        net.eval()
        device = "cuda"
        right = 0
        cnt = 0
        for i, data in enumerate(data_loader):
            data, target = data
            data = data.to(device)
            output = net(data)
            # print(output.argmax(dim=1))
            # print(target.shape)
            target = torch.argmax(target, dim=1).to(device)
            right += (output.argmax(dim=1) == target).sum().item()
            cnt += data.shape[0]
        print("cross Test accu: %4f%%." % (right / cnt * 100))
        return (right / cnt * 100)


    def main1(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        # net = MyBadNet().to(device)
        net = torch.load(save_path+'CleanModel.model')
        fun = nn.MSELoss()
        optimizier = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        train_data = datasets.MNIST(root=path + "/data/",
                                    train=True,
                                    download=False)
        train_data = MyDataset(train_data, target=1, portion=0.6, device=device)
        train_data_loader = DataLoader(dataset=train_data,
                                       batch_size=64,
                                       shuffle=True)
        data_test = datasets.MNIST(data_path, train=False, download=False,                               )
        data_test_orig = MyDataset(data_test, portion=0)
        data_test_trig = MyDataset(data_test, target=1, portion=1)
        test_loader_orig = DataLoader(data_test_orig, batch_size=64, shuffle=True)
        test_loader_trig = DataLoader(data_test_trig, batch_size=64, shuffle=True)
        # sgd = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        #######################


        max_accu = 0
        for epoch in range(10):
            # train(net, test_loader_orig, optimizer=optimizer, device=device)
            # train(net, train_data_loader, sgd, criterion=fun)
            self.train(net, train_data_loader, optimizier,)
                  # criterion=fun)
            accu1 = self.test(net, test_loader_orig)
            accu2 = self.test(net, test_loader_trig)
            if accu1+accu2 > max_accu:
                max_accu = accu1+accu2
                torch.save(net, save_path+'BadModel.model')
        tkinter.messagebox.showinfo('提示', 'Backdoor模型训练完成。\n')

    def main2(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        net = MyBadNet().to(device)
        net2 = MyBadNet().to(device)
        criterion = nn.MSELoss()
        criterion2 = nn.CrossEntropyLoss()
        optimizier = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        optimizier2 = optim.SGD(net2.parameters(), lr=0.01, momentum=0.9)
        train_data = datasets.MNIST(root=path + "/data/",
                                    train=True,
                                    download=False)
        train_data = MyDataset(train_data,  portion=0, device=device)
        train_data_loader = DataLoader(dataset=train_data,
                                       batch_size=64,
                                       shuffle=True)
        data_test = datasets.MNIST(data_path, train=False, download=False,                               )
        data_test_orig = MyDataset(data_test, portion=0)
        test_loader_orig = DataLoader(data_test_orig, batch_size=64, shuffle=True)
        #######################

        max_accu = 0
        for epoch in range(20):
            # train(net, test_loader_orig, optimizer=optimizer, device=device)
            # train(net, train_data_loader, sgd, criterion=fun)
            self.train(net, train_data_loader, optimizier, criterion)
            self.train2(net2, train_data_loader, optimizier2,criterion2)
            # criterion=fun)
            accu = self.test(net, test_loader_orig)
            accu = self.test2(net2, test_loader_orig)
            # if accu > max_accu:
            #     max_accu = accu
            #     torch.save(net, save_path+'CleanModel.model')
        # max_accu = 0.01
        tkinter.messagebox.showinfo('提示', 'Clean模型训练完成。\nAccu: %.4f' % max_accu)


    def ModelEval(self):
        device = "cuda"
        print("****************Model_Eval******************")
        net = torch.load(save_path+'CleanModel.model')
        badnet = torch.load(save_path+'BadModel.model')
        data_test = datasets.MNIST(data_path, train=False, download=False, )
        data_test_orig = MyDataset(data_test, portion=0)
        data_test_trig = MyDataset(data_test, target=1, portion=1)
        test_loader_orig = DataLoader(data_test_orig, batch_size=64, shuffle=True)
        test_loader_trig = DataLoader(data_test_trig, batch_size=64, shuffle=True)

        net_accu = self.test(net, test_loader_orig)
        net_BackdoorAccu = self.test(net, test_loader_trig)

        badnet_accu = self.test(badnet, test_loader_orig)
        badnet_BackdoorAccu = self.test(badnet, test_loader_trig)
        tkinter.messagebox.showinfo('****************Model_Eval******************',
                                    "Clean model's accu: %.4f%%\nClean model's backdoor accu: %.4f%%\nBackdoor model's accu: %.4f%%\nBackdoor model's backdoor accu: %.4f%%" % (net_accu, net_BackdoorAccu, badnet_accu, badnet_BackdoorAccu))

        print("Clean model's accu: %.4f%%" % net_accu)
        print("Clean model's backdoor accu: %.4f%%\n" % net_BackdoorAccu)
        print("Backdoor model's accu: %.4f%%" % badnet_accu)
        print("Backdoor model's backdoor accu: %.4f%%" % badnet_BackdoorAccu)


if __name__ == '__main__':
    exm = ui()
    exm.main2()


