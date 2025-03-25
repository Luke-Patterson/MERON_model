import os
import cv2
from shutil import copyfile, rmtree

photo_folder = 'C:\Users\Hussein Lightwalla\Desktop\meron_pics\marsabit_meron_survey'
rejected_photos_folder = 'C:\Users\Hussein Lightwalla\Desktop\meron_rejected_pics'
approved_photos_folder = 'C:\Users\Hussein Lightwalla\Desktop\meron_approved_pics'
#rmtree(rejected_photos_folder)
#rmtree(approved_photos_folder)
#os.mkdir(rejected_photos_folder)
#os.mkdir(approved_photos_folder)
lst_rejected_pics = []
lst_approved_pics = []
cascPath = 'C:\Users\Hussein Lightwalla\Haar-Training\Haar Training\cascade2xml\myfacedetector.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
approved_images_list = open('{}\{}'.format('C:\Users\Hussein Lightwalla\Desktop','approved_images.txt'), 'w')
rejected_images_list = open('{}\{}'.format('C:\Users\Hussein Lightwalla\Desktop','rejected_images.txt'), 'w')

for subdir, dirs, files in os.walk(photo_folder):
    for file in files:
        imagePath = os.path.join(subdir, file)
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(300, 300),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 1:
            lst_approved_pics.append(file)
            copyfile(imagePath, '{}\{}'.format(approved_photos_folder, file))
            approved_images_list.write("%s\n" % file)
            print 'approved {}'.format(file)
        else:
            lst_rejected_pics.append(file)
            copyfile(imagePath, '{}\{}'.format(rejected_photos_folder, file))
            rejected_images_list.write("%s\n" % file)
            print 'rejected {}'.format(file)

approved_images_list.close()
rejected_images_list.close()

#print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
#for (x, y, w, h) in faces:
    #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#cv2.imshow("Faces found", image)
#cv2.waitKey(0)