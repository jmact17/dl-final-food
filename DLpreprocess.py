import numpy as np
import torch
import json
from PIL import Image
from matplotlib import pyplot


def get_data(file_path, first_class, second_class):
    with open(file_path) as f:
        img_dict = json.load(f)

    file_names = [] # list of strings: all image file names
    labels = []
    for food in img_dict: # chicken or cannoli
        img_list = img_dict[food]
        for img in img_list:
            temp = img.split("/")
            name = temp[1]
            file_names.append("./images/combined/" + name + ".jpg")
            label = temp[0]
            if label == second_class:
                labels.append(1) # chicken_curry is 1
            else:
                labels.append(0) # cannoli is 0
    labels = np.array(labels)

    # loop through all image files to find min width and min height
    widths = []
    heights = []
    for f in file_names:
        img = Image.open(f)
        w, h = img.size
        widths.append(w)
        heights.append(h)
    min_width = np.amin(np.array(widths)) # 242 (train)
    min_height = np.amin(np.array(heights)) # 226 (train)

    # loop through all image files to resize & crop, and convert to 3D image matrices
    images = [] # list of matrix representations of images
    for f in file_names:
        img = Image.open(f)
        w, h = img.size
        # RESIZE THEN CROP
        if w/min_width <= h/min_height:
            if w > min_width: # resize to min width
                new_h = int(min_width * h / w)
                img = img.resize((min_width, new_h))
            if img.size[1] > min_height: # crop to min_height (crop bottom)
                img = img.crop((0, 0, min_width, min_height))
        else:
            if h > min_height: # resize to min height
                new_w = int(min_height * w / h)
                img = img.resize((new_w, min_height))
            if img.size[0] > min_width: # crop to min_width (crop right)
                img = img.crop((0, 0, min_width, min_height))
        # print("FINAL IMG SIZE", img.size)
        img.save(f)
        img_array = pyplot.imread(f)
        # print("IMG ARRAY SIZE", img_array.shape)
        images.append(img_array)

    # combine into one 4D matrix & normalize pixels
    images_matrix = (1/255.0) * np.stack(images, axis=0)
    images_matrix = np.transpose(images_matrix, (0,3,1,2))

    # print("IMAGES", images_matrix)
    # print("LABELS", labels)
    return images_matrix, labels

if __name__ == '__main__':
	get_data("meta/train.json", "cannoli", "chicken_curry")
	get_data("meta/test.json", "cannoli", "chicken_curry")
