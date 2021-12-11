import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import cv2
import mtcnn
from tensorflow.keras.models import load_model
from utils import *
from sklearn import neighbors
from sklearn import svm
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from model_openface import create_model
from Align import AlignDlib
from openface_pytorch import netOpenFace
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image

def prepare_openface():
    model = netOpenFace()
    model.load_state_dict(torch.load('openface_20180119.pth',map_location=lambda storage, loc: storage))

    return model

def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def load_feature():
    openface = prepare_openface()
    face_detector = mtcnn.MTCNN()
    people_dir = 'train2_augment'
    required_size = (160, 160)
    X = []
    y = []
    for person_name in os.listdir(people_dir):
        person_dir = os.path.join(people_dir, person_name)
        print(person_name)
        for img_name in os.listdir(person_dir):
            if img_name.endswith('.jpg') or img_name.endswith('.png'):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (160, 160))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = face_detector.detect_faces(img_rgb)
                if results:
                    res = max(results, key=lambda b: b['box'][2] * b['box'][3])
                    face, _, _ = get_face(img_rgb, res['box'])
                    face = cv2.resize(face, required_size)
                    pil_im = Image.fromarray(face)
                    transform = transforms.Compose([
                        # transforms.RandomCrop(32, padding=4),
                        # transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])

                    img_tensor = transform(pil_im)
                    img_tensor = to_var(img_tensor)
                    outputs_128, outputs_726 = openface(img_tensor)
                    outputs = to_np(outputs_128)
                    print(outputs_128)
                    print(outputs_726)
                    print(outputs)
                    # X.append(encode)
                    # y.append(person_name)

    pickle.dump(X,open('feature_openface_augment.dat','wb'))
    pickle.dump(y,open('class_openface_augment.dat','wb'))

def train_knn(X_train, X_test, y_train, y_test):
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree', weights='uniform')

    st_train_time_knn = time.time()
    knn_clf.fit(X_train, y_train)
    end_train_time_knn = time.time()

    train_time_knn = end_train_time_knn - st_train_time_knn

    st_test_time_knn = time.time()
    prediction_knn = knn_clf.predict(X_test)
    end_test_time_knn = time.time()

    test_time_knn = end_test_time_knn - st_test_time_knn
    print('Facenet KNN Result')

    print('train time: {} test time: {}'.format(train_time_knn,test_time_knn))
    print(accuracy_score(y_test, prediction_knn))
    # print(confusion_matrix(y_test,prediction_knn))
    print(classification_report(y_test,prediction_knn))

def vary_k(X_train, X_test, y_train, y_test):
    no_neighbors = np.arange(1, 40)
    train_accuracy = np.empty(len(no_neighbors))
    test_accuracy = np.empty(len(no_neighbors))

    for i, k in enumerate(no_neighbors):
        # We instantiate the classifier
        knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        # Fit the classifier to the training data
        knn.fit(X_train, y_train)

        # Compute accuracy on the training set
        train_accuracy[i] = knn.score(X_train, y_train)

        # Compute accuracy on the testing set
        test_accuracy[i] = knn.score(X_test, y_test)

    # Visualization of k values vs accuracy

    plt.title('k-NN: Varying Number of Neighbors')
    plt.plot(no_neighbors, test_accuracy, label='Testing Accuracy')
    plt.plot(no_neighbors, train_accuracy, label='Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()

def getClassifier(ktype):
    if ktype == 0:
        # Polynomial kernal
        return svm.SVC(kernel='poly', degree=8, gamma="auto")
    elif ktype == 1:
        # Radial Basis Function kernal
        return svm.SVC(kernel='rbf', gamma="auto")
    elif ktype == 2:
        # Sigmoid kernal
        return svm.SVC(kernel='sigmoid', gamma="auto")
    elif ktype == 3:
        # Linear kernal
        return svm.SVC(kernel='linear', gamma="auto")

# def train_svm(X_train, X_test, y_train, y_test):
#     # svm_clf = svm.SVC(kernel='linear', gamma="auto")
#     st_train_time = time.time()
#     svm_clf.fit(X_train, y_train)
#     end_train_time = time.time()
#
#     train_time = end_train_time - st_train_time
#
#     st_test_time = time.time()
#     prediction = svm_clf.predict(X_test)
#     end_test_time = time.time()
#
#     test_time = end_test_time - st_test_time
#     print('Facenet SVM Result')
#
#     print('train time: {} test time: {}'.format(train_time, test_time))
#     print(accuracy_score(y_test, prediction))
#     # print(confusion_matrix(y_test, prediction))
#     # print(classification_report(y_test, prediction))

if __name__ == '__main__':
    load_feature()
    # X = pickle.load(open('feature_facenet_augment.dat', 'rb'))
    # y = pickle.load(open('class_facenet_augment.dat', 'rb'))
    # X = np.array(X)
    # y = np.array(y)
    # print(X.shape)
    # print(y.shape)
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    #
    # train_svm(X_train, X_test, y_train, y_test)
    # kernels = ['Polynomial', 'RBF', 'Sigmoid', 'Linear']
    # for i in range(4):
    #     # Separate data into test and training sets
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)  # Train a SVC model using different kernal
    #     svclassifier = getClassifier(i)
    #     svclassifier.fit(X_train, y_train)  # Make prediction
    #     y_pred = svclassifier.predict(X_test)  # Evaluate our model
    #     print("Evaluation:", kernels[i], "kernel")
    #     print(accuracy_score(y_test, y_pred))
        # print(classification_report(y_test, y_pred))