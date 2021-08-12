from joblib import load
import cv2
import matplotlib.pyplot as plt
import random
from os import listdir
from os.path import isfile, join
import sys
from matplotlib.widgets import TextBox
import pandas as pd

class IdealChecker:
    def __init__(self):
        self.face_model = load('./model/face_cluster.pkl')
        self.eye_model = load('./model/eye_cluster.pkl')
        self.nose_model = load('./model/nose_cluster.pkl')
        self.lips_model = load('./model/lips_cluster.pkl')
        self.ch = []
        self.df = pd.read_csv('./csv/male_coord.csv')


    def ideal_check(self, path='./data/male/'):
        self.df = pd.read_csv('./csv/male_coord.csv')
        files = self.df['file_name']
        file = [i for i in range(len(files))]

        def key_press(event):
            if event.key == '1':
                self.ch.append(c1)
                plt.close()
            elif event.key == '2':
                self.ch.append(c2)
                plt.close()

        for i in range(10):
            fig = plt.figure(1)
            c1 = random.choice(file)
            c1_img = cv2.imread(path+files[c1], cv2.IMREAD_COLOR)
            c1_img = cv2.cvtColor(c1_img, cv2.COLOR_BGR2RGB)
            file.remove(c1)
            c2 = random.choice(file)
            c2_img = cv2.imread(path+files[c2], cv2.IMREAD_COLOR)
            c2_img = cv2.cvtColor(c2_img, cv2.COLOR_BGR2RGB)
            file.remove(c2)
            ax = fig.add_subplot(1,2,1)
            plt.imshow(c1_img)
            ax = fig.add_subplot(1,2,2)
            plt.imshow(c2_img)
            
            cid = plt.connect('key_press_event', key_press)
            plt.show()
    
    def get_ideal(self ,path='./data/male/'):
        face = [self.face_model.labels_[i] for i in self.ch]
        eye = [self.eye_model.labels_[i] for i in self.ch]
        nose = [self.nose_model.labels_[i] for i in self.ch]
        lips = [self.lips_model.labels_[i] for i in self.ch]
        
        mod_face = max(face, key=face.count)
        mod_eye = max(eye, key=eye.count)
        mod_nose = max(nose, key=nose.count)
        mod_lips = max(lips, key=lips.count)

        ideal_dict ={}
        
        for i in range(len(self.face_model.labels_)):
            if self.face_model.labels_[i]==mod_face:
                try:
                    ideal_dict[i]+=1
                except:
                    ideal_dict[i]=1
            if self.eye_model.labels_[i]==mod_eye:
                try:
                    ideal_dict[i]+=1
                except:
                    ideal_dict[i]=1
            if self.nose_model.labels_[i]==mod_nose:
                try:
                    ideal_dict[i]+=1
                except:
                    ideal_dict[i]=1
            if self.lips_model.labels_[i]==mod_lips:
                try:
                    ideal_dict[i]+=1
                except:
                    ideal_dict[i]=1
            
        img = cv2.imread(path+self.df['file_name'][max(ideal_dict, key=ideal_dict.get)], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()
    
        
        

        


        
        

checker = IdealChecker()
checker.ideal_check()
checker.get_ideal()
