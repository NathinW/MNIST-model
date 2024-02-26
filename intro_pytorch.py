import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def get_data_loader(training = True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:s
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

    train_set=datasets.FashionMNIST('./data',train=True,download=True,transform=custom_transform)
    test_set=datasets.FashionMNIST('./data', train=False,transform=custom_transform)

    

    if training == True:
        return torch.utils.data.DataLoader(train_set, batch_size = 64)
    else:
        return torch.utils.data.DataLoader(test_set, batch_size = 64)

    
def build_model():
    """
    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))

def train_model(model, train_loader, criterion, T):
    """
    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(0, T):
        running_loss = 0.0
        model.train()
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        percent = round(100*(correct/total), 2)
        print(f"Train Epoch: {epoch}   Accuracy: {correct}/{total} ({percent}%) Loss: {round(running_loss/total, 3)}")
        running_loss = 0

def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(test_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        if show_loss:
            print(f"Average loss: {round(running_loss/total, 4)}")
            print(f"Accuracy: {round(100*(correct/total), 2)}%")
        else:
            print(f"Accuracy: {round(100*(correct/total), 2)}%")

def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1
    

    RETURNS:
        None
    """
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    prob = F.softmax(model(test_images[index]), dim=1)
    values, indices = torch.topk(prob, k=3)
    values = values.tolist()[0]
    indices = indices.tolist()[0]
    value1 = values[0], indices[0]
    value2 = values[1], indices[1]
    value3 = values[2], indices[2]
    valuesList = [value1, value2, value3]
    valuesList = sorted(valuesList, key=lambda x: x[0])
    print(f"{class_names[valuesList[2][1]]}: {round(valuesList[2][0]*100, 2)}%")
    print(f"{class_names[valuesList[1][1]]}: {round(valuesList[1][0]*100, 2)}%")
    print(f"{class_names[valuesList[0][1]]}: {round(valuesList[0][0]*100, 2)}%")


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''

    criterion = nn.CrossEntropyLoss()
    model = build_model()
    
    train_model(model, get_data_loader(), criterion, 4)
    evaluate_model(model, get_data_loader(), criterion)

    images, labels = next(iter(get_data_loader(False)))
    predict_label(model, images, 1)
