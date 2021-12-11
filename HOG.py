import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import cv2
import mtcnn
# from keras.models import load_model
from utils import get_face, l2_normalizer, normalize, save_pickle, plt_show, get_encode
from sklearn import neighbors
from sklearn import svm
from skimage import feature
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def load_feature():
    people_dir = 'new_dataset'
    required_size = (160, 160)
    face_detector = mtcnn.MTCNN()
    X = []
    y = []
    for person_name in os.listdir(people_dir):
        person_dir = os.path.join(people_dir, person_name)
        print(person_name)
        person_dir = os.path.join(person_dir, 'train')
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
                    # face = normalize(face)
                    face = cv2.resize(face, required_size)
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    encode = feature.hog(face, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
                    X.append(encode)
                    y.append(person_name)

    pickle.dump(X,open('feature_HOG_augment.dat','wb'))
    pickle.dump(y,open('class_HOG_augment.dat','wb'))

def load_feature_test():
    people_dir = 'new_dataset'
    required_size = (160, 160)
    face_detector = mtcnn.MTCNN()
    X_test = []
    y_test = []
    for person_name in os.listdir(people_dir):
        person_dir = os.path.join(people_dir, person_name)
        print(person_name)
        person_dir = os.path.join(person_dir, 'train')
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
                    # face = normalize(face)
                    face = cv2.resize(face, required_size)
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    encode = feature.hog(face, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
                    X_test.append(encode)
                    y_test.append(person_name)
    return X_test,y_test

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
    print('HOG KNN Result')

    print('train time: {} test time: {}'.format(train_time_knn,test_time_knn))
    print(accuracy_score(y_test, prediction_knn))
    # print(confusion_matrix(y_test,prediction_knn))
    print(classification_report(y_test,prediction_knn, digits=4))

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
        return svm.SVC(C=10, kernel='poly', degree=1, gamma="auto")
    elif ktype == 1:
        # Radial Basis Function kernal
        return svm.SVC(C=10, kernel='rbf', gamma="auto")
    elif ktype == 2:
        # Sigmoid kernal
        return svm.SVC(C=10, kernel='sigmoid', gamma="auto")
    elif ktype == 3:
        # Linear kernal
        return svm.SVC(C=10, kernel='linear', gamma="auto")

def train_svm(X_train, X_test, y_train, y_test):
    svm_clf = svm.SVC(kernel='linear', gamma="auto")
    st_train_time = time.time()
    svm_clf.fit(X_train, y_train)
    end_train_time = time.time()

    train_time = end_train_time - st_train_time

    st_test_time = time.time()
    prediction = svm_clf.predict(X_test)
    end_test_time = time.time()

    test_time = end_test_time - st_test_time
    print('HOG SVM Result')

    print('train time: {} test time: {}'.format(train_time, test_time))
    print(accuracy_score(y_test, prediction))
    # print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction, digits=4))

if __name__ == '__main__':
    load_feature()
    # X = pickle.load(open('feature_HOG_augment.dat', 'rb'))
    # y = pickle.load(open('class_HOG_augment.dat', 'rb'))
    # X = np.array(X)
    # y = np.array(y)
    # print(X.shape)
    # print(y.shape)
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    # train_knn(X_train, X_test, y_train, y_test)

    # train_svm(X_train, X_test, y_train, y_test)
    # kernels = ['Polynomial', 'RBF', 'Sigmoid', 'Linear']
    # for i in range(4):
    #     # Separate data into test and training sets
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # Train a SVC model using different kernal
    #     svclassifier = getClassifier(i)
    #     svclassifier.fit(X_train, y_train)  # Make prediction
    #     y_pred = svclassifier.predict(X_test)  # Evaluate our model
    #     print("Evaluation:", kernels[i], "kernel")
    #     print(accuracy_score(y_test, y_pred))
    #     # print(classification_report(y_test, y_pred))