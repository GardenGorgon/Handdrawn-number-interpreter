#add your imports here
from __future__ import division
from __future__ import print_function
from sys import argv
from glob import glob
from scipy import misc
from pathlib import Path
import numpy
import random
import cv2
import math
import scipy
import scipy.misc
import os
import tensorflow as tf
import NewDataWrapper #Need this to access our custom data wrapper
"""
add whatever you think it's essential here
"""
##We put the definitions for our CNN here, and I guess we can save the path to the trained data.


class SymPred():
## This whole SymPred thing is an object that is supposed to be an identified character. Self is the name of the character, so the source images name
## followed by some identifier. prediction is what character we think it is, x1 y1 x2 and y2 are the points where we think the character begins and ends
## It's just for organizing our answers.
        def __init__(self,prediction, x1, y1, x2, y2):
                """
                <x1,y1> <x2,y2> is the top-left and bottom-right coordinates for the bounding box
                (x1,y1)
                           .--------
                           |        |
                           |        |
                            --------.
                                     (x2,y2)
                """
                self.prediction = prediction
                self.x1 = x1
                self.y1 = y1
                self.x2 = x2
                self.y2 = y2
        def __str__(self):
                return str(self.prediction) + '\t' + '\t'.join([
                                                                                                str(self.x1),
                                                                                                str(self.y1),
                                                                                                str(self.x2),
                                                                                                str(self.y2)])

class ImgPred():
        ##ImgPred is another object that is a big old list of every SymPred object we made for an entire image. Self is the ImgPreds name, image_name is
        ##The source images name, Sym_Pred_List is the list of simpreds, latex is for extra credit stuff, aka what the equation is in LATEX form.
        def __init__(self,image_name,sym_pred_list,latex = 'LATEX_REPR'):
                """
                sym_pred_list is list of SymPred
                latex is the latex representation of the equation
                """
                self.image_name = image_name
                self.latex = latex
                self.sym_pred_list = sym_pred_list
        def __str__(self):
                res = self.image_name + '\t' + str(len(self.sym_pred_list)) + '\t' + self.latex + '\n'
                for sym_pred in self.sym_pred_list:
                        res += str(sym_pred) + '\n'
                return res

def predict(image_path, wrapper):


        """
        Add your code here
        """
        """
        #Don't forget to store your prediction into ImgPred
        img_prediction = ImgPred(...)
        """

        #This is where all of our code will probably go. Here are the steps to success

        
        #Step One: Make a list which will contain the locations of every character in our source Image.
        SymPredList = []

        #Step Two: Go down that list we just made and use the code from PA4 in conjunction with our new Model to analyze each character. George made this part.
        #This is the find a character part of the code. Max and George worked it out.
        im = cv2.imread(image_path,0)
        (thresh, imbw) = cv2.threshold(im,20,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #cv2.imwrite('clapfuck.jpg', imbw)
        im3,contours,hierarchy = cv2.findContours(imbw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        idx = 0
        for cnt in contours:
                idx += 1
                x1,y1,w,h = cv2.boundingRect(cnt)
                roi=imbw[y1:y1+h,x1:x1+w]

                #Step Two.1: Make a Numpy Array of all the pixels starting from the top left corner of an identified character to the bottom right corner of the identified character.
                height, width = roi.shape
                if height >= width:
                        padded = cv2.copyMakeBorder(roi,0,0,(height-width)//2,(height-width)//2,cv2.BORDER_CONSTANT,value=[0,0,0])
                else:
                        padded = cv2.copyMakeBorder(roi,(width-height)//2,(width-height)//2,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
                Smol = cv2.resize(padded, (28, 28))
                (thresh, evaluateMe) = cv2.threshold(Smol, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                #scipy.misc.imsave(os.path.basename(file), ree)
                #Step Two.2: Feed that numpy into our PA4 image analyzer converter thing but using our new trained model
                evaluateMeMe = numpy.reshape(evaluateMe, (1, 28, 28, 1))
                prediction = tf.argmax(y_conv,1)
                final_number = prediction.eval(feed_dict={x:evaluateMeMe,y_:numpy.zeros((1,40)), keep_prob:1.0})
                #Step Two.3: Record what we think it is as the prediction field of the SymPred we are currently on
                final_guess = wrapper.label_types[int(final_number)]
                DisSymPred = SymPred(final_guess,x1,y1,x1+w,y1-h)
                SymPredList.append(DisSymPred)

        #Step Three: Wrap that now complete SymPred list, in an ImgPred, fill out all the fields of that ImgPred, and then return that shit.
        img_prediction = ImgPred(os.path.basename(image_path), SymPredList)

        #Step Four: Were Donezo
        return img_prediction



if __name__ == '__main__':
        #First, we have to load the data structures needed for the CNN. This is done with the tutorial.
        x = tf.placeholder(tf.float32, [None, 28, 28, 1]) #placeholder for input images
        y_ = tf.placeholder(tf.float32, [None, 40]) #placeholder for output value (0-39)
        sess = tf.InteractiveSession() #This sets up an interactive Tensorflow Python session
        sess.run(tf.global_variables_initializer())
        #Here is the actual convolutional network layering process
        #First Layer
        def weight_variable(shape): #Define Weights
          initial = tf.truncated_normal(shape, stddev=0.1)
          return tf.Variable(initial)
        def bias_variable(shape): #Define Bias
          initial = tf.constant(0.1, shape=shape)
          return tf.Variable(initial)
        def conv2d(x, W):
          return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        def max_pool_2x2(x):
          return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                      strides=[1, 2, 2, 1], padding='SAME')
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        x_image = tf.reshape(x, [-1,28,28,1])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        #Second Layer
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        #Densely Connected Layer
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        #Dropout
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        #Readout Layer
        W_fc2 = weight_variable([1024, 40])
        b_fc2 = bias_variable([40])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        #I'm relatively sure this shit is just the method our grading TA will call
        #We don't have to do anything here, except maybe ask our user for the path to the trained data

        wrapper = NewDataWrapper.DigitsData()


        my_folder = input ("Please input the absolute path for the folder in which you have a trained model: ")
        my_path = Path(my_folder + "\\checkpoint")
        #this makes the path to our trained network if we have one, or to where we will save the training data
        if my_path.is_file():
                print("Trained data found. \n")
                #if we have trained data, we restore the old trained model and take in input
                saver = tf.train.Saver()
                saver.restore(sess, my_folder + "\\model.ckpt") #Here we restore our trained data.
                #Now we ask the user to input a folder path...
                #...and use glob.glob to extract individual images into a list of paths
                import glob
                image_folder_path = input ("Please input an absolute path for the folder containing test/image data: ")

                isWindows_flag = False
                if len(argv) == 3:
                        isWindows_flag = True
                if isWindows_flag:
                        image_paths = glob.glob(image_folder_path + '\\*png')
                else:
                        image_paths = glob.glob(image_folder_path + '/*png')
                results = []
                for image_path in image_paths:
                        impred = predict(image_path, wrapper)
                        results.append(impred)

                with open(image_folder_path +"/predictions.txt","w") as fout:
                        for res in results:
                                fout.write(str(res))
                        fout.close

                

        else:
                print("Trained Data not found.")
