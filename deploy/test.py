import face_model
import argparse
import cv2
import sys
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/model-r100-ii/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()


model = face_model.FaceModel(args)
img = cv2.imread('52.jpg')
img = model.get_input(img)
f1 = model.get_feature(img)

print(type(f1))
print(f1.shape)
print(f1[0:10])

#gender, age = model.get_ga(img)
#print(gender)
#print(age)
#sys.exit(0)
img = cv2.imread('394.jpg')
img = model.get_input(img)
f2 = model.get_feature(img)
dist = np.sum(np.square(f1-f2), 0)
print("distance:", dist**0.5)
sim = np.dot(f1, f2.T)
print("similarity", sim)
#diff = np.subtract(source_feature, target_feature)
#dist = np.sum(np.square(diff),1)
