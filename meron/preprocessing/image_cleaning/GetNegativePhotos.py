#based on a list in a text file, if a file isnt in there, push it to a negative folder.
import os
from shutil import copyfile

folder_negative_pics = 'C:\Users\Hussein Lightwalla\Haar-Training\Haar Training\{}\{}'.format('training', 'negative')
folder_positive_pics = 'C:\Users\Hussein Lightwalla\Haar-Training\Haar Training\{}\{}\{}'.format('training', 'positive', 'rawdata')

lst_goodpics = []
with open('C:\Users\Hussein Lightwalla\Haar-Training\Haar Training\{}\{}\{}'.format('training', 'positive', 'info.txt')) as f:
    for line in f:
        line = line[8:]
        index = line.index('.')
        line = line[:index]
        lst_goodpics.append(line)
        #print line

for filename in os.listdir(folder_positive_pics):
    imagePath = os.path.join(folder_positive_pics, filename)
    index = filename.index('.')
    filename_tocheck = filename[:index]
    if filename_tocheck not in lst_goodpics:
        copyfile(imagePath, '{}\{}'.format(folder_negative_pics, filename))
        os.remove(imagePath)
        print filename