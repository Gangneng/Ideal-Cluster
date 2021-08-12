from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import face_alignment

class PreProcessing:
    # png 파일을 jpg로 변환하는 함수입니다.
    def png2jpg(file_list, path="./data/male/"):
        for i in file_list:
            if 'jpg' not in i and 'jpeg' not in i and 'JPG' not in i and 'JPEG' not in i:
                im = Image.open(path+i)
                rgb_im = im.convert('RGB')
                os.remove(path+i)
                rgb_im.save(path+i.split('.')[0]+'.jpg')

    # 연예인 이름 | 파일 이름
    # csv 파일을 생성합니다.
    # 사용자에게 결과를 보여줄 때 활용될 예정입니다.
    def to_csv(file_list, path="./data/male/"):
        # 미리 저장된 csv 파일을 가져옵니다. 없으면 새롭게 pandas DataFrame을 활용하여 생성해주도록 합니다.
        data = []
        try:
            if 'male' in path:
                data = pd.read_csv('./csv/male.csv')
            else:
                data = pd.read_csv('./csv/female.csv')
        except:
            data = pd.DataFrame(columns=['name','file_name'])

        # 파일 이름은 0000.jpg, 0001.jpg의 형식으로 지정됩니다.
        r_cnt = len(data) # csv를 불러오고 난 후의 행 개수
        for i in file_list:
            name, im_format = i.split('.')[0], i.split('.')[1]
            file_name = str('{0:04d}'.format(r_cnt))+'.'+im_format
            try:
                name = int(name)
            except:
                os.rename(path+i, path+file_name)
                data.loc[r_cnt]=[name, file_name]
                r_cnt+=1

        if 'male' in path:
            data.to_csv('./csv/male.csv', mode='w', index=False)
        else:
            data.to_csv('./csv/female.csv', mode='w', index=False)

    def get_coord(file_list, path="./data/male/"):
        df = pd.DataFrame(columns=['file_name', 'face','eyebrow1','eyebrow2','eye1','eye2','nose','lips','total'])

        face_detector = 'sfd'
        face_detector_kwargs = {
            "filter_threshold" : 0.8
        }

        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
            face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)

        for i in files:
            try:
                print(i+" get coords start.")
                preds = fa.get_landmarks(path+i)[-1]

                face = preds[0:17]
                eyebrow1 = preds[17:22]
                eyebrow2 = preds[22:27]
                nose = preds[27:31]
                eye1 = preds[36:42]
                eye2 = preds[42:48]
                lips = preds[48:60]

                df.loc[len(df)]=[i, face, eyebrow1, eyebrow2, eye1, eye2, nose, lips, preds]
            except RuntimeError:
                print(i+' occured Error')

        if 'male' in path:
            df.to_csv('./csv/male_coord.csv', mode='w', index=False)
        else:
            df.to_csv('./csv/female_coord.csv', mode='w', index=False)

PPobj = PreProcessing
files = [f for f in listdir('./data/male') if isfile(join('./data/male',f))]

#PPobj.png2jpg(files)
#PPobj.to_csv(files)
#PPobj.get_coord(files)