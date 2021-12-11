import matplotlib as plt
import numpy as np
from HOG import *
from facenet import *
from LBPH import *
from openface import *

def load_all():
    X_HOG = pickle.load(open('feature_HOG_augment.dat', 'rb'))
    y_HOG = pickle.load(open('class_HOG_augment.dat', 'rb'))
    X_facenet = pickle.load(open('feature_facenet_augment.dat', 'rb'))
    y_facenet = pickle.load(open('class_facenet_augment.dat', 'rb'))
    X_LBPH = pickle.load(open('feature_LBPH_augment.dat', 'rb'))
    y_LBPH = pickle.load(open('class_LBPH_augment.dat', 'rb'))
    # X_HOG = pickle.load(open('feature_HOG_augment.dat', 'rb'))
    # y_HOG = pickle.load(open('class_HOG_augment.dat', 'rb'))

    return X_HOG,y_HOG,X_LBPH,y_LBPH,X_facenet,y_facenet

def train(X, y):
    no_neighbors = np.arange(1, 40)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
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

    return test_accuracy

def knn_line_plot(facenet, hog,lbph):
    no_neighbors = np.arange(1, 40)

    plt.title('k-NN: Varying Number of Neighbors')
    plt.plot(no_neighbors, facenet, label='facenet Accuracy')
    plt.plot(no_neighbors, hog, label='HOG Accuracy')
    plt.plot(no_neighbors, lbph, label='LBPH Accuracy')
    # plt.plot(no_neighbors, openface, label='openface Accuracy')
    # plt.plot(no_neighbors, train_accuracy, label='Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == '__main__':
    X_HOG,y_HOG,X_LBPH,y_LBPH,X_facenet,y_facenet = load_all()

    facenet = train(X_facenet,y_facenet)
    HOG = train(X_HOG,y_HOG)
    LBPH = train(X_LBPH,y_LBPH)

    knn_line_plot(facenet,HOG,LBPH)