import numpy as np
import torch
import json
from PIL import Image
from matplotlib import pyplot


def get_data(file_path, first_class, second_class):
    """
    Inputs:
    Outputs: 4D matrix of images
    """
    # store train and test data in to lists
    with open(file_path) as f:
        img_dict = json.load(f)

    file_names = [] # list of strings: all image file names
    labels = []
    for food in img_dict: # chicken or cannoli
        img_list = img_dict[food] # list of cannoli and chicken curry
        for img in img_list:
            temp = img.split("/")
            name = temp[1]
            # print("item is", item)
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
    min_width = np.amin(np.array(widths)) # 242
    min_height = np.amin(np.array(heights)) # 226

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
        print("IMG ARRAY SIZE", img_array.shape)
        images.append(img_array)

    # combine into one 4D matrix & normalize pixels
    images_matrix = (1/255.0) * np.stack(images, axis=0)

    # shuffle images and labels in same order **do this in train or here??**
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    labels = torch.tensor(labels[indices])
    images_matrix = torch.tensor(images_matrix[indices,:,:,:])
    images_matrix = torch.reshape(images_matrix, (images_matrix.shape[0],images_matrix.shape[3],images_matrix.shape[1],images_matrix.shape[2]))

    # one hot labels?
    #print("IMAGES", images_matrix)
    #print("LABELS", labels)
    print("images matrix size", images_matrix.shape)
    return images_matrix, labels

if __name__ == '__main__':
	get_data("meta/train.json", "cannoli", "chicken_curry")
	get_data("meta/test.json", "cannoli", "chicken_curry")
    # SHUFFLING EXAMPLE
    # indices = np.arange(10)
    # np.random.shuffle(indices)
    # x = np.arange(10)
    # print(x)
    # x = x[indices]
    # print(x)
# premade pytorch implementation:
model = torch.hub.load('pytorch/vision:v0.4.2', 'googlenet', pretrained=True)
model.eval()
input_batch =
output = model(input_batch)
print(output[0])
print(torch.nn.functional.softmax(output[0], dim=0))




# #Google stuff:
# if __name__ == "__main__":
#     img = imageio.imread('cat.jpg', pilmode='RGB')
#     img = np.array(Image.fromarray(img).resize((224, 224))).astype(np.float32)
#     img[:, :, 0] -= 123.68
#     img[:, :, 1] -= 116.779
#     img[:, :, 2] -= 103.939
#     img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
#     img = img.transpose((2, 0, 1))
#     img = np.expand_dims(img, axis=0)
