from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
import numpy as np
from django.views.decorators.csrf import csrf_exempt
from sentence_transformers import SentenceTransformer #loading bert sentence model
from keras.models import load_model
import multiprocessing
import re
from playsound import playsound
import cv2
import soundfile
import librosa
import subprocess
from keras.preprocessing.image import img_to_array
import base64
from django.core.files.base import ContentFile
from keras.models import model_from_json
from django.core.files.storage import FileSystemStorage

text_emotion_labels = ['sad', 'happy', 'neutral', 'angry', 'scared', 'surprise']
bert = SentenceTransformer('nli-distilroberta-base-v2')
global player

detection_model_path = 'model/haarcascade_frontalface_default.xml'
emotion_model_path = 'model/_mini_XCEPTION.106-0.65.hdf5'
face_detection = cv2.CascadeClassifier(detection_model_path)
EMOTIONS = ['angry','disgust','scared','happy','neutral','sad','surprise']

speech_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'scared' 'disgust', 'surprise', 'unknown']

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    sound_file.close()        
    return result

def DetectEmotion(request):
    if request.method == 'POST':
        name = request.POST.get('t1', False)
        output = checkEmotion()
        context= {'data':output}
        return render(request, 'index.html', context)
    
def WebCam(request):
    if request.method == 'GET':
        data = str(request)
        formats, imgstr = data.split(';base64,')
        imgstr = imgstr[0:(len(imgstr)-2)]
        data = base64.b64decode(imgstr)
        if os.path.exists('EmotionApp/static/photo/test.png'):
            os.remove('EmotionApp/static/photo/test.png')
        with open('EmotionApp/static/photo/test.png', 'wb') as f:
            f.write(data)
        f.close()
        context= {'data':"done"}
        return HttpResponse("Image saved")

def PlaySong(request):
    if request.method == 'GET':
        global player
        name = request.GET.get('t1', False)
        arr = name.split("-")
        print(arr)
        player = multiprocessing.Process(target=playsound, args=("songs/"+arr[0]+"/"+arr[1],))
        player.start()
        output = '<table border="1" align="center"><tr><th>Click here to stop player</th></tr>'     
        output += '<td><a href=\'StopSound?data=stop\'><font size=3 color=black>Click Here to Stop</font></a></td></tr>'
        output += '</table></body></html>'
        context= {'data':output}
        return render(request, 'index.html', context)  

def StopSound(request):
    if request.method == 'GET':
        global player
        player.terminate()
        output = "Audio Stopped Successfully"
        context= {'data':output}
        return render(request, 'index.html', context)

def getRecommendation(predict):
    songs = '<table border="1" align="center"><tr><th>Song File</th><th>Click here to play</th></tr>'
    for root, dirs, directory in os.walk('songs/'+predict):
        for j in range(len(directory)):
            songs += '<tr><td>'+directory[j]+'</td><td><a href=\'PlaySong?t1='+predict+"-"+directory[j]+'\' target="blank"><font size=3 color=black>Play Song</font></a></td></tr>'
    return songs+'</table><br/>'        
    



def VideoChatbot(request):
    if request.method == 'GET':
        return render(request, 'VideoChatbot.html', {})
 

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def checkEmotion():
    global label
    #emotion_classifier = load_model(emotion_model_path, compile=False)
    with open('model/cnnmodel.json', "r") as json_file:
        loaded_model_json = json_file.read()
        emotion_classifier = model_from_json(loaded_model_json)
    json_file.close()    
    emotion_classifier.load_weights("model/cnnmodel_weights.h5")
    emotion_classifier._make_predict_function()                  
    orig_frame = cv2.imread('EmotionApp/static/photo/test.png')
    frame = cv2.imread('EmotionApp/static/photo/test.png',0)
    faces = face_detection.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    print("==================="+str(len(faces)))   
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        roi = orig_frame[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (32, 32))
        roi = roi.reshape(1,32,32,3)
        roi = roi.astype('float32')
        img = roi/255
        #preds = emotion_classifier.predict(roi)[0]
        #emotion_probability = np.max(preds)
        preds = emotion_classifier.predict(img)
        predict = np.argmax(preds)
        label = EMOTIONS[predict]
        output = 'Predicted Emotion = <font size="3" color="blue">'+label+'</font>'
        output += getRecommendation(label)
    return output   
        

