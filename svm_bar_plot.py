import matplotlib.pyplot as plt
import numpy as np

def svm_bar_plot(number_X_axis, facenet, HOG,LBPH):
    N = number_X_axis
    ind = np.arange(N)
    width = 0.2
    plt.bar(ind, facenet, width, label='facenet')
    plt.bar(ind + (width), HOG, width,label='HOG')
    plt.bar(ind + (2*width), LBPH, width, label='LBPH')

    plt.ylabel('accuracy')
    plt.title('SVM model performance')

    plt.xticks(ind + 0.3, ('Polynomial', 'RBF','Sigmoid','Linear'))
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    facenet = (0.739,0.971,0.934,0.984)
    HOG = (0.743, 0.890, 0.743, 0.962)
    LBPH = (0.797, 0.854, 0.777, 0.796)
    svm_bar_plot(4, facenet, HOG,LBPH)