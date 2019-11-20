#batch data
import numpy as np
#import pytorch as torch
import json


def get_data(file_path, first_class, second_class):
    # store train and test data in to lists
    with open(file_path) as f:
        train_dict = json.load(f)

    inputs = []
    labels = []
    for food in train_dict: #for chicken or cannoli
        img_list = train_dict[food]#list of cannoli and chicken curry
        for img in img_list:
            temp = img.split("/")
            item = temp[1]
            # print("item is", item)
            inputs.append(item)
            label = temp[0]
            if label == second_class:
                labels.append(1)
            else:
                labels.append(0) #cannoli is 0
    #print("trainLabelsList", trainLabelsList)

    # shuffle list
    np.random.shuffle(labels)
    print("labels is", labels)

if __name__ == '__main__':
	get_data("train.json", "cannoli", "chicken_curry") #cat/dog
	#get_data("CIFAR_data_compressed/test", 3, 5)
# def get_data(file_path, first_class, second_class)
#     #unpickled_file = unpickle(file_path)
# 	#inputs = unpickled_file[b'data'] #images
# 	#labels = unpickled_file[b'labels']
# 	labelsnew = []
# 	inputsnew = []
# 	for input, label in zip(inputs,labels):
# 		if label == first_class or label == second_class:
# 			labelsnew.append(label)
# 			inputsnew.append(input)
#
# 	#reshape inputs
# 	inputsnew = np.asarray(inputsnew).astype(np.float32)/255.0 #or np.asarray(inputsnew)
# 	reshaped_inputs = np.reshape(inputsnew, (-1, 3, 224, 224)) #3 channels (RGB), 224 pixels image width, 224 pixels image height
# 	transposed_inputs = np.transpose(reshaped_inputs, [0, 2, 3, 1])
#
# 	#changing 5->0, (dog) what is the label for cat?
# 	labelsnew = [1 if label == second_class else 0 for label in labelsnew]
#
# 	#turn labels into one-hot vectors
# 	labelsnew = tf.one_hot(labelsnew, depth=2)
# 	return transposed_inputs, labelsnew
# #
# #
#
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
