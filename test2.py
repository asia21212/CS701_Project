import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import cv2
import mtcnn
from keras.models import load_model
from utils import get_face, l2_normalizer, normalize, save_pickle, plt_show, get_encode
from sklearn import neighbors
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

encoder_model = 'facenet_keras.h5'
people_dir = 'train2'

recognition_t = 0.3
required_size = (160, 160)
encoding_dict = dict()


face_detector = mtcnn.MTCNN()
face_encoder = load_model(encoder_model, compile=False)
X = []
y = []

for person_name in os.listdir(people_dir):
    person_dir = os.path.join(people_dir, person_name)
    encodes = []
    for img_name in os.listdir(person_dir):
        if img_name.endswith('.jpg') or img_name.endswith('.png'):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (160,160))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detector.detect_faces(img_rgb)
            if results:
                res = max(results, key=lambda b: b['box'][2] * b['box'][3])
                face, _, _ = get_face(img_rgb, res['box'])

                face = normalize(face)
                face = cv2.resize(face, required_size)
                encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
                # encodes.append(encode)
                X.append(encode)
                y.append(person_name)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
knn_clf = neighbors.KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree', weights='uniform')
knn_clf.fit(X_train, y_train)

prediction = knn_clf.predict(X_test)
print(accuracy_score(y_test, prediction))
print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))

error_rate = []

# for i in range(1, 40):
#     knn = neighbors.KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train, y_train)
#     pred_i = knn.predict(X_test)
#
#     error_rate.append(np.mean(pred_i != y_test))
#
# plt.figure(figsize=(10, 6))
#
# plt.plot(range(1, 40), error_rate, color='blue', linestyle='--',
#              markersize=10, markerfacecolor='red', marker='o')
# plt.title('K versus Error rate')
# plt.xlabel('K')
# plt.ylabel('Error rate')
# plt.show()

# train_accuracy = np.empty(len(neighbors))
# test_accuracy = np.empty(len(neighbors))
#
# # Loop over different values of k
# for i, k in enumerate(neighbors):
#     # Setup a k-NN Classifier with k neighbors: knn
#     knn = KNeighborsClassifier(n_neighbors=k)
#
#     # Fit the classifier to the training data
#     knn.fit(X_train, y_train)
#
#     # Compute accuracy on the training set
#     train_accuracy[i] = knn.score(X_train, y_train)
#
#     # Compute accuracy on the testing set
#     test_accuracy[i] = knn.score(X_test, y_test)
#
# # Generate plot
# plt.title('k-NN: Varying Number of Neighbors')
# plt.plot(2, test_accuracy, label='Testing Accuracy')
# plt.plot(2, train_accuracy, label='Training Accuracy')
# plt.legend()
# plt.xlabel('Number of Neighbors')
# plt.ylabel('Accuracy')
# plt.show()

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