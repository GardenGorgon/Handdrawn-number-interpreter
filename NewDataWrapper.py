import numpy as np
import random
from scipy import misc
import scipy
import cv2
import os
#from pathlib import Path
import glob

class DigitsData(object):
        annotated_path = "annotated"
        #This is the folder of the images we will be using
        names_train = list()
        names_validation = list()
        labels_train = list()
        labels_validation = list()
        prev_idx = 0 #for batch iteration
        label_types = list() #list of types of labels
        id_key = list() #integer ids to correspond with the label_types list
        def __init__(self):
                file_names = os.listdir(self.annotated_path)
                file_list = glob.glob(self.annotated_path + '/*.png')
                read_pictures = list()#we need to read the images into numpy arrays
                file_labels = list() #List of all the file labels
                for file in file_list:
                    groups = file.split('_') #Split the image name into groups
                    if file.count("_", 0, len(file_list)-1) > 2: #if it is not an equation image
                        file_labels.append(groups[3])
                        read_pictures.append(scipy.misc.imread(file,1))
                        #now we save the type of label if applicable
                        if groups[3] not in self.label_types:
                            self.label_types.append(groups[3])
                            self.id_key.append(len(self.label_types))
                #Now we have to split each list into two parts: training and validation
                for index in range(0, len(read_pictures)):
                    if index > len(read_pictures)/5:
                        self.names_train.append(read_pictures[index])
                        type_index = self.label_types.index(file_labels[index])
                        self.labels_train.append(self.id_key[type_index])
                    else:
                        self.names_validation.append(read_pictures[index])
                        type_index = self.label_types.index(file_labels[index])
                        self.labels_validation.append(self.id_key[type_index])

        def get_valid(self,size):
                #size should be 697
                valid_name_matrix = np.zeros((size, 28,28,1)) #Create matrices
                valid_label_matrix = np.zeros((size,40))
                prev_indx = 0
                for i in range (size): #store lists into them
                        valid_name_matrix[i,:,:,0] = self.names_validation[i]
                        valid_label_matrix[i,self.labels_validation[i]-1] = 1
                #Return a tuple with the two matrices
                return valid_name_matrix, valid_label_matrix
        def shuffle(self):
                pass
        def next_batch(self,batch_size):
                #Batch size should be 64
                train_name_matrix = np.zeros((batch_size,28,28,1))#create batch matrices
                train_label_matrix = np.zeros((batch_size, 40))
                for i in range(batch_size):#make the actual batches
                        if self.prev_idx+i >=len(self.names_train):
                                self.prev_idx = 0
                        train_name_matrix[i,:,:,0] = self.names_train[self.prev_idx+i]
                        train_label_matrix[i,self.labels_train[self.prev_idx+i]-1] = 1
                self.prev_idx = self.prev_idx + batch_size
                #Return a tuple with the two matrices
                return (train_name_matrix, train_label_matrix)
