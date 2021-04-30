import cv2

import os
import glob
current = os.getcwd()
new_path = os.path.join(current,"final dataset")
print(new_path)
try:
    os.mkdir(new_path)
except:
    print("Already exist")
for filename in glob.glob('*.jpg'):
    src = cv2.imread(os.path.join(current,filename), cv2.IMREAD_UNCHANGED)
    print(src.shape)
    #percent by which the image is resized
    scale_percent = 15


    #calculate the 50 percent of original dimensions
    width = 512
    height = 251
    # dsize

    dsize = (width, height)

    # resize image
    output = cv2.resize(src, dsize)

    cv2.imwrite(os.path.join(new_path,filename),output)
