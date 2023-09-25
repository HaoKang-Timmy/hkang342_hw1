import torch
import torchvision
def create_dataset():

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                torchvision.transforms.Normalize((0.5,), (0.5,))])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    X_train = train_dataset.data.view(-1, 28*28).numpy()
    y_train = train_dataset.targets.numpy()

    X_test = test_dataset.data.view(-1, 28*28).numpy()
    y_test = test_dataset.targets.numpy()

    return X_train, y_train, X_test, y_test