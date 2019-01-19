import torch
import torchvision.datasets as dsets
import numpy as np
import random
import copy
import time
from svrg import SVRG
from data import load_mnist


#Get MNIST using autograd data loader
#Couldn't download MNIST on my system with pytorch built in dataloader
N, train_images, train_labels, test_images,  test_labels = load_mnist()

#Convert training data from Numpy arrays to pytorch tensors
x, y= torch.from_numpy(train_images), torch.from_numpy(train_labels)
x,y = x.type(torch.FloatTensor),y.type(torch.FloatTensor)

#Convert training data from Numpy arrays to pytorch tensors
x_test, y_test = torch.from_numpy(test_images), torch.from_numpy(test_labels)
x_test, y_test = x_test.type(torch.FloatTensor),y_test.type(torch.FloatTensor)

#MLP dimensions
D_in, H1, H2, D_out = 784, 200, 100, 10

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.ReLU(),
    torch.nn.Linear(H1,H2),
    torch.nn.ReLU(),
    torch.nn.Linear(H2,D_out)
    )

model2 = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.ReLU(),
    torch.nn.Linear(H1,H2),
    torch.nn.ReLU(),
    torch.nn.Linear(H2,D_out)
    )

##model2 = copy.deepcopy(model) #Use same initial weights for both networks

loss_fn = torch.nn.CrossEntropyLoss()

alpha = 1e-2
freq = 100 #how often to recompute large gradient
            #The optimizer will not update model parameters on iterations
            #where the large batches are calculated

lg_batch = 3000 #size of large gradient batch 
min_batch = 300 #size of mini batch

optimizer = SVRG(model.parameters(), lr = alpha, freq = freq)
optimizer2 = torch.optim.SGD(model2.parameters(), lr = alpha)

epochs = 50
iterations = int (epochs * (60000/min_batch))

#SVRG Training
counter = 0
total = time.time()
while(counter < iterations):
    #compute large batch gradient
    temp = np.random.choice(x.size()[0], lg_batch, replace = False) #calculate batch indices
    indices = torch.from_numpy(temp)
    temp2 = torch.index_select(x, 0, indices)
    y_pred = model(temp2)
    yt = torch.index_select(y, 0, indices).type(torch.LongTensor)
    loss = loss_fn(y_pred, torch.max(yt, 1)[1])
    optimizer.zero_grad()
    loss.backward()
    counter+=1
    optimizer.step()

    #update models using mini batch gradients
    for i in range(freq-1):
        temp = np.random.choice(x.size()[0], min_batch, replace = False)
        indices = torch.from_numpy(temp)
        temp2 = torch.index_select(x, 0, indices)
        y_pred = model(temp2)
        yt = torch.index_select(y, 0, indices).type(torch.LongTensor)
        loss = loss_fn(y_pred, torch.max(yt, 1)[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        counter += 1
        if (counter == iterations):
            break
print('time for SVRG ' + str(iterations) +' steps')
print(time.time()-total)
print('')

#SGD Training
total = time.time()
for t in range(iterations):
    temp = np.random.choice(x.size()[0], min_batch, replace = False)
    indices = torch.from_numpy(temp)
    temp2 = torch.index_select(x, 0, indices)
    y_pred = model2(temp2)
    yt = torch.index_select(y, 0, indices).type(torch.LongTensor)
    loss = loss_fn(y_pred, torch.max(yt, 1)[1])
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()
print('time for SGD ' + str(iterations) +' steps')
print(time.time()-total)
print('')

#print train accuracy SVRG
y_predn = model(x).data.numpy()
yn = y.data.numpy()
pred = np.argmax(y_predn, axis = 1)
goal = np.argmax(yn, axis = 1)
acc = np.sum(pred == goal)/60000
print('train acc SVRG')
print(acc)
print('')

#print train accuracy SGD
y_predn = model2(x).data.numpy()
yn = y.data.numpy()
pred = np.argmax(y_predn, axis = 1)
goal = np.argmax(yn, axis = 1)
acc = np.sum(pred == goal)/60000
print('train acc SGD')
print(acc)
print('')



#print test accuracy SVRG
y_predn = model(x_test).data.numpy()
yn = y_test.data.numpy()
pred = np.argmax(y_predn, axis = 1)
goal = np.argmax(yn, axis = 1)
acc = np.sum(pred == goal)/10000
print('test acc SVRG')
print(acc)
print('')

#print test accuracy SGD
y_predn = model2(x_test).data.numpy()
yn = y_test.data.numpy()
pred = np.argmax(y_predn, axis = 1)
goal = np.argmax(yn, axis = 1)
acc = np.sum(pred == goal)/10000
print('test acc SGD')
print(acc)


