from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
from skimage import exposure
from numpy import expand_dims
import os, random, time, imutils

print("Starting the Process..")

def brightness_augmentation(sam, filename, img):

    b1 = random.uniform(0.3, 0.5)
    b2 = random.uniform(0.7, 0.9)

    imageDataGenerator_obj = ImageDataGenerator(brightness_range=[b1, b2])
    iterator = imageDataGenerator_obj.flow(sam, batch_size=1)

    for i in range(5):
        chunk = iterator.next()
        sub_img = chunk[0].astype('uint8')
        cv2.imwrite(filename.split('.')[0] + "_brightness_"+str(i)+".jpg", sub_img)

def horizontal_flip(sam, filename):

    imageDataGenerator_obj = ImageDataGenerator(horizontal_flip = True)
    iterator = imageDataGenerator_obj.flow(sam, batch_size=1)

    sub_img = iterator.next()[0].astype('uint8')
    cv2.imwrite(filename.split('.')[0] + "_hFlip.jpg", sub_img)



def contrast_stretching(sam, filename, img):
    for i in  range(5):
        p2 = int(random.randrange(0, 5))
        p98 = int(random.randrange(75, 98))
        p2, p98 = np.percentile(img, (p2, p98))
        img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
        cv2.imwrite(filename.split('.')[0] + "_contrast_"+str(i)+".jpg", img_rescale)

def rotate_images(img,filename):

    for angle in np.arange(0, 360, 90):
        rotated = imutils.rotate_bound(img, angle)
        cv2.imwrite(filename.split('.')[0] + "_roatated_"+str(angle)+".jpg", rotated)




def data_augmentation(dir):

    data_path = dir
    dataset = os.path.abspath(data_path)
    users = os.listdir(dataset)

    if not users:
        print("No Data is present in the Dataset Folder!")
        print("Quitting..")
        quit()

    for user in users:
        user_path = dataset + "/" + user + "/"
        user_files = os.listdir(user_path)


        if not user_files:
            print("No Data is present for the user {}". format(user))
            continue

        os.chdir(user_path)


        try:
            f = open("."+user, "x")
            f.close()
            print("Creating Data for {}".format(user))
        except:
            print("Data is already augmented for the user {}". format(user))
            print("Skipping... \n")
            continue





        for file in user_files:
            if file.endswith('.jpg'):
                img = cv2.imread(file)
                img_array = img_to_array(img)

                # dimension adjustment
                sam = expand_dims(img_array, 0)
                brightness_augmentation(sam, file, img)
                contrast_stretching(sam, file, img)


        print("Brightness and Contrast Done!")
        time.sleep(1)

        for file in os.listdir(user_path):
            if file.endswith('.jpg'):

                img = cv2.imread(file)
                img_array = img_to_array(img)

                sam = expand_dims(img_array, 0)
                horizontal_flip(sam, file)
                # rotate_images(img, file)

        print("Horizontal Flip and Roate is Done! \n\n")

    print("Completed for All Users!")


if __name__ == '__main__':
    data_augmentation()