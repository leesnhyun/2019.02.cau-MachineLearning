import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import time

transform = transforms.Compose([transforms.Resize((100,100)),
                                transforms.Grayscale(),		# the code transforms.Graysclae() is for changing the size [3,100,100] to [1, 100, 100] (notice : [channel, height, width] )
                                transforms.ToTensor(),])

size_row	= 100    # height of the image
size_col  	= 100   # width of the image

train_num=1027
test_num=256
batch=3
#train_data_path = 'relative path of training data set'
train_data_path = './horse-or-human/train'
trainset = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
# change the valuse of batch_size, num_workers for your program
# if shuffle=True, the data reshuffled at every epoch
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=False, num_workers=1)


validation_data_path = './horse-or-human/validation'
valset = torchvision.datasets.ImageFolder(root=validation_data_path, transform=transform)
# change the valuse of batch_size, num_workers for your program
valloader = torch.utils.data.DataLoader(valset, batch_size=3, shuffle=False, num_workers=1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def Loss_Func(label,sigm,num):
    return (1/num)*np.sum(Funtion(label,sigm))

def Funtion(label,sigm):
    return -(np.nan_to_num(label*np.log(sigm.flatten())) + np.nan_to_num((1-label)*np.log(1-sigm.flatten())))


def Accuracy(label, sigm, num):
    #     print(len(sigm),"len(sigm)")
    #     count =0
    #     for i in range(len(sigm)):
    #         if(sigm[i]>0.5):
    #             sigm[i]=1
    #             if(sigm[i]-label[i]==0):
    #                 count=count+1

    #         else:
    #             sigm[i]=0
    #             if(sigm[i]-label[i]==0):
    #                 count=count+1
    #     return count / batch

    #     cor = 0
    #     for i in range(num):
    #         if (sigm[i] > 0.5 and label[i] == 1):
    #             cor = cor + 1
    #         elif (sigm[i] <= 0.5 and label[i] == 0):
    #             cor = cor + 1
    #     return cor / num

    sigm = sigm.reshape(-1)
    arr = np.array(list(map(lambda x: 1 if x >= 0.5 else 0, sigm)))
    arrs = list(filter(lambda x: x == 0, arr - label))
    return len(arrs) / num

def M(w,data):
    return np.dot(w,data)

#a,b,c,v,w,train_matrix,
def d_sigmoid(m):
    return sigmoid(m)*(1-sigmoid(m))

#a,b,c,v,w,train_matrix,
def d_u(a,b,c,x,v,w,label):
#     temp1=sigmoid(c)-label
#     temp2=w*d_sigmoid(b)
#     temp3=v*d_sigmoid(a)
#     temp4=x
#     temp=np.dot(np.dot(np.dot(temp1,temp2),temp3),temp4.T)
    temp=sigmoid(c)-label
    temp=np.dot(w.T,temp)
    temp=temp*d_sigmoid(b)
    temp=np.dot(v.T,temp)
    temp=temp*d_sigmoid(a)
    temp=np.dot(temp,x.T)
    return (1/len(a[0]))* temp

#c,b,a,train_matrix,v,w
def d_v(a,b,c,w,label):

    temp=(sigmoid(c)-label)
    temp=np.dot(w.T,temp)
    temp=temp*d_sigmoid(b)
    temp=np.dot(temp,sigmoid(a).T)
    return (1/len(b[0]))* temp

#a,b,c,v,w,train_matrix,
def d_w(b,c,label):
    temp=np.dot((sigmoid(c)-label),sigmoid(b).T)
    return (1/len(c[0]))*temp


train_matrix = np.zeros(((size_col * size_row + 1), train_num), dtype=float)
train_matrix_count = 0
train_label = np.zeros(train_num, dtype=float)

NUM_EPOCH = 1
for epoch in range(NUM_EPOCH):
    # load training images of the batch size for every iteration
    for i, data in enumerate(trainloader):

        # inputs is the image
        # labels is the class of the image
        inputs, labels = data

        # if you don't change the image size, it will be [batch_size, 1, 100, 100]
        inputs = np.array(inputs)
        labels = np.array(labels)

        for i in range(len(inputs)):
            for j in range((size_col * size_row)):
                train_matrix[j][train_matrix_count + i] = inputs[i, 0, int(j / 100), int(j % 100)]
            train_matrix[(size_col * size_row)][train_matrix_count + i] = 1
            train_label[train_matrix_count + i] = labels[i]
        train_matrix_count = train_matrix_count + len(inputs)
print(train_matrix)

test_matrix = np.zeros(((size_col * size_row + 1), test_num), dtype=float)
test_matrix_count = 0
test_label = np.zeros(test_num, dtype=float)

NUM_EPOCH = 1
for epoch in range(NUM_EPOCH):
    # load testing images of the batch size for every iteration
    for i, data in enumerate(valloader):

        # inputs is the image
        # labels is the class of the image
        inputs, labels = data

        # if you don't change the image size, it will be [batch_size, 1, 100, 100]
        inputs = np.array(inputs)
        labels = np.array(labels)

        for i in range(len(inputs)):
            for j in range((size_col * size_row)):
                test_matrix[j][test_matrix_count + i] = inputs[i, 0, int(j / 100), int(j % 100)]
            test_matrix[(size_col * size_row)][test_matrix_count + i] = 1
            test_label[test_matrix_count + i] = labels[i]
        test_matrix_count = test_matrix_count + len(inputs)

train_varLoss = []
train_varAccuracy = []

test_varLoss = []
test_varAccuracy = []

times = []
# make 2 hidden layer with k1 , k2
k1 = 10
k2 = 5

# make hidden layer's matrix
u = np.random.randn(k1, 10001)
v = np.random.randn(k2, k1)
w = np.random.randn(1, k2)

# determine the leaning rate
p = 0.05

past = 0
while_count = 0
while True:
    # 변수들과의 곱
    start = time.time()

    a_train = M(u, train_matrix)
    b_train = M(v, a_train)
    c_train = M(w, b_train)

    a_test = M(u, test_matrix)
    b_test = M(v, a_test)
    c_test = M(w, b_test)

    Der_w = d_w(b_train, c_train, train_label)
    Der_v = d_v(a_train, b_train, c_train, w, train_label)
    Der_u = d_u(a_train, b_train, c_train, train_matrix, v, w, train_label)

    w = w - p * Der_w
    v = v - p * Der_v
    u = u - p * Der_u

    end = time.time()

    train_Loss = Loss_Func(train_label, sigmoid(c_train), train_num)
    train_varLoss.append(train_Loss)
    print(train_Loss)
    test_Loss = Loss_Func(test_label, sigmoid(c_test), test_num)
    test_varLoss.append(test_Loss)

    train_Accu = Accuracy(train_label, sigmoid(c_train), train_num)
    train_varAccuracy.append(train_Accu)
    test_Accu = Accuracy(test_label, sigmoid(c_test), test_num)
    test_varAccuracy.append(test_Accu)

    times.append(end - start)

    while_count = while_count + 1

    if abs(past - train_Loss) < 10e-6 and while_count > 20:
        print("finish")
        break
    else:
        past = train_Loss
        continue

plt.title("Loss")
plt.xlabel("itoration")
plt.ylabel("Loss")
plt.plot(np.arange(1, len(train_varLoss) + 1), train_varLoss,color='red',alpha=0.5)
plt.plot(np.arange(1, len(test_varLoss) + 1), test_varLoss,color='blue',alpha=0.5)
plt.show()

plt.title("Accuracy")
plt.xlabel("itoration")
plt.ylabel("accuracy")
plt.plot(np.arange(1, len(train_varAccuracy) + 1), train_varAccuracy,color='red',alpha=0.5)
plt.plot(np.arange(1, len(test_varAccuracy) + 1), test_varAccuracy,color='blue',alpha=0.5)
plt.show()

print(" | loss | accuracy |")
print("---------------------------------------")
print("training | %.2f | %.2f |" % (train_varLoss[-1], train_varAccuracy[-1]))
print("---------------------------------------")
print("validation | %.2f | %.2f |" % (test_varLoss[-1], test_varAccuracy[-1]))
print("---------------------------------------")

