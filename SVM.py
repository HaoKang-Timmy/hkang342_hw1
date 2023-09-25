import torch
import torchvision
from torchvision import transforms
# from sklearn import svm
from SVMClassifier import SVM
from sklearn.metrics import accuracy_score

# Load and preprocess the MNIST dataset using PyTorch
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
X_train = mnist_dataset.data.view(-1, 28 * 28).float()
y_train = mnist_dataset.targets.numpy()

mnist_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
X_test = mnist_test_dataset.data.view(-1, 28 * 28).float()
y_test = mnist_test_dataset.targets.numpy()

# Linear Kernel SVM
linear_svm = SVM(kernel='linear')
linear_svm.fit(X_train, y_train)

# RBF Kernel SVM
rbf_svm = SVM(kernel='rbf')
rbf_svm.fit(X_train, y_train)

# Predictions
print("begin predict")
linear_train_pred = linear_svm.predict(X_train)
rbf_train_pred = rbf_svm.predict(X_train)

linear_test_pred = linear_svm.predict(X_test)
rbf_test_pred = rbf_svm.predict(X_test)

# Evaluate Linear SVM
linear_train_accuracy = accuracy_score(y_train, linear_train_pred)
linear_test_accuracy = accuracy_score(y_test, linear_test_pred)

print("Linear SVM Accuracy (Train): {:.2f}%".format(linear_train_accuracy * 100))
print("Linear SVM Accuracy (Test): {:.2f}%".format(linear_test_accuracy * 100))

# Evaluate RBF SVM
rbf_train_accuracy = accuracy_score(y_train, rbf_train_pred)
rbf_test_accuracy = accuracy_score(y_test, rbf_test_pred)

print("RBF SVM Accuracy (Train): {:.2f}%".format(rbf_train_accuracy * 100))
print("RBF SVM Accuracy (Test): {:.2f}%".format(rbf_test_accuracy * 100))