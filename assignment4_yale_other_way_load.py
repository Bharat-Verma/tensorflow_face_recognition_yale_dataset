#!/usr/bin/python
"""
Created on Tue May  2 15:49:46 2017

@author: bverma
"""

import os, numpy as np
#from scipy import ndimage
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt 

image_size = 224  # Pixel width and height.
pixel_depth = 256.0  # Number of levels per pixel.
num_channels = 1 # 3 for colored images 
#seed = 42



folder_training = '/home/bverma/Documents/tensorflow_work/CroppedYale_resized_pad_224/training_all'
folder_testing = '/home/bverma/Documents/tensorflow_work/CroppedYale_resized_pad_224/testing_all'


image_folder_train = sorted(os.listdir(folder_training))

image_folder_test = sorted(os.listdir(folder_testing))

#np.random.RandomState(seed)
#np.random.shuffle(image_folder_train)
#np.random.shuffle(image_folder_test)
#print (len(image_folder_train))

# Function to plot images 
def plot_images(images, cls_true, cls_pred = None):
    
    assert len(images) == len(cls_true) == 10
    fig, axes = plt.subplots(5,2)
    fig.subplots_adjust(hspace = 0.3, wspace = 0.3)
    
    for i, ax in enumerate(axes.flat):
        
        ax.imshow(images[i].reshape(image_size, image_size) ,cmap = 'binary')
        
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, pred: {1}".format(cls_true[i],cls_pred[i])
            
        ax.set_xlabel(xlabel)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show()

def read_img(path):
    img = cv2.imread(path,0)
    img = img.astype(dtype=np.float32)/256.0
    return img


def load_differently_as_tuple_training():
    directory = '/home/bverma/Documents/tensorflow_work/CroppedYale_resized_pad_224/training_all'
    data = {0:{}}
    labels = []
    seed = 22345 #seed for the random state
    #read the images from the directory
    for subdir, dirs, files in os.walk(directory):
        for image in files:
            r = read_img(directory+'/'+image)
            s = image.split('_')
            #adding this line for yale            
            s = s[0].split('B')
            if int(s[1]) not in labels:
                labels.append(int(s[1]))

            try:
                data[0][int(s[1])].append([r,int(s[1])])
            except KeyError:
                data[0][int(s[1])] = [[r,int(s[1])]]
   
    np.random.RandomState(seed)
    train_set = list(); 
    #give the real label to the list
    for k,v in data.iteritems():
        for k2, v2 in v.iteritems():
            temp = []
            for item in data[k][k2]:
                temp.append((item[0],labels.index(item[1])))
            np.random.shuffle(temp)    
            data[k][k2] = temp
            for image_tuple in data[k][k2][:]:
                train_set.append(image_tuple)    
                
    np.random.RandomState(seed)
    np.random.shuffle(train_set)
    print (len(train_set))
    return train_set  

def load_differently_as_tuple_testing():
    directory = '/home/bverma/Documents/tensorflow_work/CroppedYale_resized_pad_224/testing_all'
    data = {0:{}}
    labels = []
    seed = 22345 #seed for the random state
    #read the images from the directory
    for subdir, dirs, files in os.walk(directory):
        for image in files:
            r = read_img(directory+'/'+image)
            s = image.split('_')
            #adding this line for yale            
            s = s[0].split('B')
            if int(s[1]) not in labels:
                labels.append(int(s[1]))

            try:
                data[0][int(s[1])].append([r,int(s[1])])
            except KeyError:
                data[0][int(s[1])] = [[r,int(s[1])]]

    np.random.RandomState(seed)
    test_set = list()
    #give the real label to the list
    for k,v in data.iteritems():
        for k2, v2 in v.iteritems():
            temp = []
            for item in data[k][k2]:
                temp.append((item[0],labels.index(item[1])))
            np.random.shuffle(temp)    
            data[k][k2] = temp
            for image_tuple in data[k][k2][:]:
                test_set.append(image_tuple)
                
    np.random.RandomState(seed)
    np.random.shuffle(test_set) 
    print (len(test_set))
    return test_set 


#def load_training_data(folder_training, image_folder):
#  num_images = 0
#  dataset_class = np.ndarray(shape = (len(image_folder)), dtype = np.int32)
#  
#  """Load the data for a single letter label."""
#  for image_name in image_folder:
#      img = cv2.imread((folder_training+"/"+image_name))
#      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#      #img = np.reshape(img, (img.shape[0], img.shape[1], 1))
#      
#      img = img.astype(float)/256.0
##      cv2.imshow('image',img)
##      cv2.waitKey(0)
#      dataset = np.ndarray(shape=(len(image_folder), image_size, image_size),
#                             dtype=np.float32)
#      image_name = image_name.split('_')[0]
#      labels = int(image_name.split('B')[1])
#      dataset_class[num_images] = int(labels-1)
#      
#      try:
#          dataset[num_images, :, :] = img
#          num_images = num_images + 1
#      except IOError as e:
#          print('Could not read:', image_name, ':', e, '- it\'s ok, skipping.')
#      
#  print('Full dataset tensor:', dataset.shape)
#  return dataset,dataset_class 

train_set = load_differently_as_tuple_training()
test_set = load_differently_as_tuple_testing()

training_data = np.asarray([image_tuple[0] for image_tuple in train_set])
training_labels = np.asarray([image_tuple[1] for image_tuple in train_set]) 

testing_data = np.asarray([image_tuple[0] for image_tuple in test_set])
testing_labels = np.asarray([image_tuple[1] for image_tuple in test_set])

training_data = training_data.reshape(-1,224,224,1)  
testing_data  = testing_data.reshape(-1,224,224,1)
#overall_dataset_training,overall_class_training = load_training_data(folder_training, image_folder_train)

#overall_dataset_testing,overall_class_testing = load_training_data(folder_testing,image_folder_test)

#num_classes = len(np.unique(overall_class_training))
num_classes = len(np.unique(training_labels))
#num_classes = 37

# One hot encoding - Converting into labels

def reformat(dataset_class):
    
    dataset_class = (np.arange(num_classes) == dataset_class[:, None]).astype(np.float32)
    
    return dataset_class

#one hot encoded data
#train_labels = reformat(overall_class_training)
#test_labels = reformat(overall_class_testing)

train_labels = reformat(training_labels)
test_labels = reformat(testing_labels)

##########################################################################



#Plot sample images
#images = overall_dataset_training[0:]
#cls_true = overall_class_training[0:]
#plot_images(images,cls_true)


img_size_flat = image_size * image_size
image_shape = (image_size, image_size)
num_channels = 1
num_classes = 38

#convolution layer 1
filter_size1 = 5
num_filters1 = 16

#convolution layer 2
filter_size2 = 5
num_filters2 = 32

#fully connected layer
fc_size = 128

#weights function
def new_weights(shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    return tf.Variable(initializer(shape=shape))
    
#biases function
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))
    
#convoulutional layer build
def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling = True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    
    layer = tf.nn.conv2d(input = input,
                         filter = weights,
                         strides = [1,1,1,1],
                         padding = 'SAME')

    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value = layer,
                               ksize = [1,2,2,1],
                               strides = [1,2,2,1],
                               padding = 'SAME')                         

    layer = tf.nn.relu(layer)
    
    return layer, weights
    
#flatten layer to flatten the conv layer
def flatten_layer(layer):
    layer_shape = layer.get_shape()
    
    num_features = layer_shape[1:4].num_elements()
    
    layer_flat = tf.reshape(layer, [-1, num_features])
    
    return layer_flat, num_features
    
#building new fully connected layer
def new_fc_layer(input, num_inputs, num_outputs, use_relu = 'TRUE'):
    weights = new_weights(shape = [num_inputs, num_outputs])
    biases = new_biases(length = num_outputs)
    layer = tf.matmul(input, weights) + biases
    
    if use_relu:
        layer = tf.nn.relu(layer)
    
    return layer
    
#placeholders
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name = 'x' )

x_image = tf.reshape(x, shape = [-1, image_size, image_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name = 'y_true')

y_true_cls = tf.argmax(y_true, dimension = 1)


#create convolutional layer 1
conv_layer1, weights_conv1 = new_conv_layer(input = x_image,
                             num_input_channels = num_channels,
                             filter_size = filter_size1,
                             num_filters = num_filters1,
                             use_pooling = 'TRUE')

#create convolutional layer 2
conv_layer2, weights_conv2 = new_conv_layer(input = conv_layer1,
                             num_input_channels = num_filters1,
                             filter_size = filter_size2,
                             num_filters = num_filters2,
                             use_pooling = 'TRUE')  
                             
#flatten layer 
flat_layer, num_features = flatten_layer(conv_layer2)    

#build fully connected layer 1
fc_layer1 = new_fc_layer(input=flat_layer,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu='TRUE')                       

#build fully connected layer 2
fc_layer2 = new_fc_layer(input=fc_layer1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu='FALSE') 
                         
#Predicted classes
y_pred = tf.nn.softmax(fc_layer2)  

y_pred_class = tf.argmax(y_pred, dimension = 1) 

#cost function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = fc_layer2,
                                                        labels = y_true)
                                                        
cost = tf.reduce_mean(cross_entropy)

#optimizer
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)

#performance measure
correct_prediction = tf.equal(y_pred_class, y_true_cls) #check this step

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                                               

################################################################

session = tf.Session()

session.run(tf.global_variables_initializer())

batch_size = 38

testing_data = testing_data.reshape(-1,50176)
feed_dict_test = {x: testing_data[:,:],
                  y_true: test_labels[:,:]}
                  
def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    
    # Print the accuracy.
    print("Accuracy on testing-set: {0:>6.1%}".format(acc))

def optimize():
    i = 0
    for epoch in range(1000):
        print ("epoch :"+str(epoch+1))
        for i in range(0,1444,38):

            x_batch = training_data[i:i+38,:,:]
            print (i)
            x_batch = x_batch.reshape(-1,50176)
            y_true_batch = train_labels[i:i+38,:]
            

            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch}
    

            session.run(optimizer, feed_dict=feed_dict_train)
            acc_train = session.run(accuracy, feed_dict=feed_dict_train)
            print("Accuracy on training-set: {0:>6.1%}".format(acc_train))

        print_accuracy()

#optimize()


train_batch_size = 38

#Initalize the iteration variable
total_iterations = 0
# Function to run the optimizer
def optimize_two(num_iterations):
    
    global total_iterations
    
    for iterations in range(total_iterations, total_iterations + num_iterations):
        
        offset = (iterations * train_batch_size) % (train_labels.shape[0] - train_batch_size)
        
        batch_data = training_data[offset : (offset + train_batch_size),:]
        #batch_data = batch_data.reshape(-1,50176)
        batch_labels = train_labels[offset : (offset + train_batch_size),:]
        
        feed_dict_train = {x_image: batch_data, y_true: batch_labels }
                
        session.run(optimizer, feed_dict=feed_dict_train)
        
        
        if (iterations % 38 == 0):
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(iterations + 1, acc))
            
            
    total_iterations += num_iterations

optimize_two(14440)


test_batch_size = 38

def print_test_accuracy(show_example_errors = False,
                        show_confusion_matrix = False):
    
    
    num_test = 418
    
    cls_pred = np.zeros(shape = num_test, dtype = np.int)
    
    i = 0
    
    while i < num_test:
        
        j = min(i+ test_batch_size , num_test)
        
        images = testing_data[i:j, :]
        images = images.reshape(-1,224,224,1)
        
        labels = test_labels[i:j, :]
        
        feed_dict = {x_image: images,
                     y_true: labels}
        
        cls_pred[i:j] = session.run(y_pred_class, feed_dict = feed_dict)
        
        i = j
        
    cls_true = testing_labels
        
    correct = (cls_true == cls_pred)
        
    correct_sum = correct.sum()
        
    acc = float(correct_sum) / num_test
        
    msg = "Accuracy on Test-set: {0: .1%} ({1}/ {2})"
        
    print(msg.format(acc, correct_sum, num_test))


#def testing_acc():        
#    i = 0
#    for i in range(0,418,38):
#    
#        x_batch = testing_data[i:i+38,:,:]
#        print (i)
#        x_batch = x_batch.reshape(-1,50176)
#        y_true_batch = test_labels[i:i+38,:]
#        
#    
#        feed_dict_test = {x: x_batch,
#                           y_true: y_true_batch}
#    
#    
#        session.run(optimizer, feed_dict=feed_dict_test)
#        acc_train = session.run(accuracy, feed_dict=feed_dict_test)
#        print("Accuracy on testing-set: {0:.1%}".format(acc_train))



    
#testing_acc()
print_test_accuracy()
