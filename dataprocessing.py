import cv2

import os
import glob
# current = os.getcwd()
current = r"C:\Users\Lucky\PycharmProjects\symmetry\data\wudi"
new_path = os.path.join(current,"final dataset")
print(new_path)
try:
    os.mkdir(new_path)
except:
    print("Already exist")
for filename in glob.glob(r'C:\Users\Lucky\PycharmProjects\symmetry\data\wudi\*.jpg'):
    src = cv2.imread(os.path.join(current,filename), cv2.IMREAD_UNCHANGED)
    print("zheshi",src)
    print(src.shape)
    #percent by which the image is resized
    scale_percent = 15

    #calculate the 50 percent of original dimensions
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(src, dsize)

    cv2.imwrite(os.path.join(new_path,filename),output) 