import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import re
import matplotlib.pyplot as plt
import cv2
import pickle
from joblib import dump

class ClusterModel:
    def __init__(self):
        self.df = ''
        # visualize를 위하여
        self.y_pred = ''
        self.face_arr = ''
        self.eyebrow_arr = ''
        self.eye_arr = ''
        self.nose_arr = ''
        self.lips_arr = ''

    def get_data(self, path='./csv/male_coord.csv'):
        self.df = pd.read_csv(path)
        for i in range(len(self.df)):
            for j in ['face','eyebrow1','eyebrow2','eye1','eye2','nose','lips','total']:
                self.df.loc[i,j]=re.sub(r"[^0-9.]","",self.df.loc[i,j])
                self.df.loc[i,j]=np.array(self.df.loc[i,j].split('.')[:-1]).astype(int)
                #self.df.loc[i,j]=np.array(self.df.loc[i,j]).astype(int)


    def KMeans(self):
        #####################
        ##  Total Cluster  ##
        #####################
        '''
        total = self.df.loc[:,'total']
        total_arr = []

        for i in total:
            total_arr.append(i)

        total_arr = np.array(total_arr)
        std_scaler = StandardScaler()
        scaled_total = std_scaler.fit_transform(total_arr)

        model = KMeans(init="k-means++",n_clusters=3, random_state=42)
        model.fit(scaled_total)
        self.y_pred = model.labels_
        '''
        #####################
        ##  face Cluster  ##
        #####################
        
        face = self.df.loc[:,'face']
        self.face_arr = []

        for i in face:
            self.face_arr.append(i)

        self.face_arr = np.array(self.face_arr)
        std_scaler = StandardScaler()
        scaled_face = std_scaler.fit_transform(self.face_arr)

        model = KMeans(init="k-means++",n_clusters=3)
        model.fit(scaled_face)
        self.y_pred = model.labels_
        
        #Kmeans 적용
        model = KMeans(init="k-means++",n_clusters=6)
        model.fit(scaled_face)
        self.y_pred = model.labels_

        #모델 저장
        joblib_file = './model/face_cluster.pkl'
        dump(model, joblib_file)

        #######################
        ##  eyebrow Cluster  ## ----- 눈썹이 이상형에 있어서 큰 의미가 없을 것 같아서 당분간은 포함시키지 않고 해보자.
        #######################
        '''
        eyebrow = self.df.loc[:,['eyebrow1','eyebrow2']]
        self.eyebrow_arr = []

        for i,j in zip(eyebrow['eyebrow1'],eyebrow['eyebrow2']):
            self.eyebrow_arr.append(np.concatenate((i,j), axis=None))
        self.eyebrow_arr = np.array(self.eyebrow_arr)
        std_scaler = StandardScaler()
        scaled_eyebrow = std_scaler.fit_transform(self.eyebrow_arr)

        model = KMeans(init="k-means++",n_clusters=3)
        model.fit(scaled_eyebrow)
        self.y_pred = model.labels_
        '''
        ###################
        ##  eye Cluster  ##
        ###################
        
        eye = self.df.loc[:,['eye1','eye2']]
        self.eye_arr = []

        for i,j in zip(eye['eye1'],eye['eye2']):
            self.eye_arr.append(np.concatenate((i,j), axis=None))
        self.eye_arr = np.array(self.eye_arr)
        std_scaler = StandardScaler()
        scaled_eye = std_scaler.fit_transform(self.eye_arr)

        #Kmeans 적용
        model = KMeans(init="k-means++",n_clusters=6)
        model.fit(scaled_eye)
        self.y_pred = model.labels_

        #모델 저장
        joblib_file = './model/eye_cluster.pkl'
        dump(model, joblib_file)

        ####################
        ##  nose Cluster  ##
        ####################
        
        nose = self.df.loc[:,'nose']
        self.nose_arr = []

        for i in nose:
            self.nose_arr.append(i)
        self.nose_arr = np.array(self.nose_arr)
        std_scaler = StandardScaler()
        scaled_nose = std_scaler.fit_transform(self.nose_arr)

        model = KMeans(init="k-means++",n_clusters=3)
        model.fit(scaled_nose)
        self.y_pred = model.labels_
        
        #Kmeans 적용
        model = KMeans(init="k-means++",n_clusters=6)
        model.fit(scaled_nose)
        self.y_pred = model.labels_

        #모델 저장
        joblib_file = './model/nose_cluster.pkl'
        dump(model, joblib_file)

        ####################
        ##  lips Cluster  ##
        ####################
        
        lips = self.df.loc[:,'lips']
        self.lips_arr = []

        for i in lips:
            self.lips_arr.append(i)
        self.lips_arr = np.array(self.lips_arr)
        std_scaler = StandardScaler()
        scaled_lips = std_scaler.fit_transform(self.lips_arr)

        model = KMeans(init="k-means++",n_clusters=5)
        model.fit(scaled_lips)
        self.y_pred = model.labels_
        
        #Kmeans 적용
        model = KMeans(init="k-means++",n_clusters=6)
        model.fit(scaled_lips)
        self.y_pred = model.labels_

        #모델 저장
        joblib_file = './model/lips_cluster.pkl'
        dump(model, joblib_file)


    def visualize_img(self):
        #######################
        ##  Total Visualize  ##
        #######################
        '''
        n = 10
        fig = plt.figure(1)
        box_index = 1
        for cluster in range(3):
            result = np.where(self.y_pred ==cluster)
            for i in np.random.choice(result[0].tolist(), n, replace=False):
                ax = fig.add_subplot(n, n, box_index)
                img = cv2.imread('./data/male/'+self.df.loc[i,'file_name'], cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                box_index+=1
        plt.show()
        '''
        ######################
        ##  Face Visualize  ##
        ######################
        '''
        n = 10
        fig = plt.figure(1)
        box_index = 1
        for cluster in range(3):
            result = np.where(self.y_pred ==cluster)
            for i in np.random.choice(result[0].tolist(), n, replace=False):
                ax = fig.add_subplot(n, n, box_index)
                img = cv2.imread('./data/male/'+self.df.loc[i,'file_name'], cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cropped = img[min(self.face_arr[i][1::2])-10:max(self.face_arr[i][1::2])+10, min(self.face_arr[i][::2])-10:max(self.face_arr[i][::2])+10]
                try:
                    plt.imshow(cropped)
                except:
                    plt.imshow(img)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                box_index+=1
        plt.show()
        '''
        #####################
        ##  eye Visualize  ##
        #####################
        '''
        n = 10
        fig = plt.figure(1)
        box_index = 1
        for cluster in range(6):
            result = np.where(self.y_pred ==cluster)
            for i in np.random.choice(result[0].tolist(), n, replace=False):
                ax = fig.add_subplot(n, n, box_index)
                img = cv2.imread('./data/male/'+self.df.loc[i,'file_name'], cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cropped = img[min(self.eye_arr[i][1::2])-10:max(self.eye_arr[i][1::2])+10, min(self.eye_arr[i][::2])-10:max(self.eye_arr[i][::2])+10]
                try:
                    plt.imshow(cropped)
                except:
                    plt.imshow(img)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                box_index+=1
        plt.show()
        ''' 
        #########################
        ##  eyebrow Visualize  ## 
        #########################
        '''
        n = 10
        fig = plt.figure(1)
        box_index = 1
        for cluster in range(3):
            result = np.where(self.y_pred ==cluster)
            for i in np.random.choice(result[0].tolist(), n, replace=False):
                ax = fig.add_subplot(n, n, box_index)
                img = cv2.imread('./data/male/'+self.df.loc[i,'file_name'], cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cropped = img[min(self.eyebrow_arr[i][1::2])-30:max(self.eyebrow_arr[i][1::2])+30, min(self.eyebrow_arr[i][::2])-30:max(self.eyebrow_arr[i][::2])+30]
                try:
                    plt.imshow(cropped)
                except:
                    plt.imshow(img)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                box_index+=1
        plt.show()
        '''      
        ######################
        ##  nose Visualize  ##
        ######################
        '''
        n = 10
        fig = plt.figure(1)
        box_index = 1
        for cluster in range(3):
            result = np.where(self.y_pred ==cluster)
            for i in np.random.choice(result[0].tolist(), n, replace=False):
                ax = fig.add_subplot(n, n, box_index)
                img = cv2.imread('./data/male/'+self.df.loc[i,'file_name'], cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cropped = img[min(self.nose_arr[i][1::2])-30:max(self.nose_arr[i][1::2])+30, min(self.nose_arr[i][::2])-30:max(self.nose_arr[i][::2])+30]
                try:
                    plt.imshow(cropped)
                except:
                    plt.imshow(img)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                box_index+=1
        plt.show()
        '''
        ######################
        ##  lips Visualize  ##
        ######################
        '''
        n = 10
        fig = plt.figure(1)
        box_index = 1
        for cluster in range(5):
            result = np.where(self.y_pred ==cluster)
            for i in np.random.choice(result[0].tolist(), n, replace=False):
                ax = fig.add_subplot(n, n, box_index)
                img = cv2.imread('./data/male/'+self.df.loc[i,'file_name'], cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cropped = img[min(self.lips_arr[i][1::2])-30:max(self.lips_arr[i][1::2])+30, min(self.lips_arr[i][::2])-30:max(self.lips_arr[i][::2])+30]
                try:
                    plt.imshow(cropped)
                except:
                    plt.imshow(img)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                box_index+=1
        plt.show()
        '''

model = ClusterModel()
model.get_data()
model.KMeans()
#model.visualize_img()