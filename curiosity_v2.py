#!/usr/bin/env python3

"""
Created on Wed Mar 13 16:09:21 2019

@author: alexandre
"""
def mail():
    from email.mime.multipart import MIMEMultipart
    from email.mime.image import MIMEImage
    from email.mime.text import MIMEText
    import smtplib
     
    # create message object instance
    msg = MIMEMultipart()
     
    # setup the parameters of the message
    password = "**********"
    msg['From'] = "alexandre75.daniel@gmail.com"
    msg['To'] = "alexandre75.daniel@gmail.com"
    msg['Subject'] = "Intrusion"
     
    # attach image to message body
    msg.attach(MIMEImage(open("intrusion.jpg", "rb").read()))
     
    # create server
    server = smtplib.SMTP('smtp.gmail.com: 587')
    server.starttls()
     
    # Login Credentials for sending the mail
    server.login(msg['From'], password)
     
    # send the message via the server.
    try:
        server.sendmail(msg['From'], msg['To'], msg.as_string())
    except:
        print("echec envoi mail")
     
    server.quit()


def quitter():
    global quitter
    quitter=True
    import sys
    sys.exit()

#-------------affichage caméra-----------

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

import threading
 
class show_webcam(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.running = False
 
    def run(self):
        global quitter
        global photo
        global photo_prise
        global zoom
        global video
        global video_prise
       # initialize the camera and grab a reference to the raw camera capture
        camera = PiCamera()
        #camera.annotate_text = 'Curiosity s eye'
        #camera.resolution = (640, 480)
        camera.resolution = (1024, 768)
        camera.framerate = 32
        #rawCapture = PiRGBArray(camera, size=(640, 480))
        rawCapture = PiRGBArray(camera, size=(1024, 768))

      # allow the camera to warmup
        time.sleep(0.1)
        
        self.running = True
        # scrutation du buffer d'entree 
        while self.running:
            #retourne l'image
            camera.rotation = 180
            for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                #enregistre une photo sur le bureau
                if photo==True:
                    photo_prise+=1
                    camera.capture('image_' + str(photo_prise) + '.jpg')
                    photo=False
                
                if zoom==True:
                    camera.zoom = (0.25, 0.25, 0.5, 0.5)
                if zoom==False:
                    camera.zoom=(0,0,1,1)
                    
                if video==True:
                    video_prise+=1
                    camera.start_recording('video_' + str(video_prise) + '.h264')
                    #Recording duration / duree enregistrement (15s)
                    camera.wait_recording(5)
                    camera.stop_recording()
                    video=False
                    
                # grab the raw NumPy array representing the image, then initialize the timestamp
                # and occupied/unoccupied text
                image = frame.array

                #Retourne l'image
                #image = cv2.flip(image, -1)
                # show the frame
                cv2.imshow("Curiosity Camera", image)
                key = cv2.waitKey(1) & 0xFF

                # clear the stream in preparation for the next frame
                rawCapture.truncate(0)
                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break
                if self.running==False:
                    break
                if quitter==True:
                    break
            self.running=False
        camera.close()
        cv2.destroyAllWindows()
 
    def stop(self):
        self.running = False
 
#fonction de gestion du thread d'affichage de la cam
def show(th,pi_cam):   
    if th==True:
        print("lancement caméra")
        pi_cam=show_webcam()
        pi_cam.start()
    if th==False:
        print("Arrêt caméra")
        pi_cam.stop()
        pi_cam.join()
    return(pi_cam)

def photo():
    print("photo")
    global photo
    photo=True
   
def zoom_in():
    print("zoom avant")
    global zoom
    zoom=True
    
def zoom_out():
    print("zoom arrière")
    global zoom
    zoom=False

def video():
    print("enregistrement vidéo")
    global video
    video=True


#---------------- Détection d'objets -- yolo---------
import sys, os, cv2, time
import numpy as np, math
from argparse import ArgumentParser
from openvino.inference_engine import IENetwork, IEPlugin

m_input_size = 416

yolo_scale_13 = 13
yolo_scale_26 = 26
yolo_scale_52 = 52

classes = 80
coords = 4
num = 3
anchors = [10,14, 23,27, 37,58, 81,82, 135,169, 344,319]

LABELS = ("person", "bicycle", "car", "motorbike", "aeroplane",
          "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird",
          "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack",
          "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat",
          "baseball glove", "skateboard", "surfboard","tennis racket", "bottle",
          "wine glass", "cup", "fork", "knife", "spoon",
          "bowl", "banana", "apple", "sandwich", "orange",
          "broccoli", "carrot", "hot dog", "pizza", "donut",
          "cake", "chair", "sofa", "pottedplant", "bed",
          "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven",
          "toaster", "sink", "refrigerator", "book", "clock",
          "vase", "scissors", "teddy bear", "hair drier", "toothbrush")

label_text_color = (255, 255, 255)
label_background_color = (125, 175, 75)
box_color = (255, 128, 0)
box_thickness = 1

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-d", "--device", help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
                                                Sample will look for a suitable plugin for device specified (CPU by default)", default="MYRIAD", type=str)
    return parser


def EntryIndex(side, lcoords, lclasses, location, entry):
    n = int(location / (side * side))
    loc = location % (side * side)
    return int(n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc)


class DetectionObject():
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    class_id = 0
    confidence = 0.0

    def __init__(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
        self.xmin = int((x - w / 2) * w_scale)
        self.ymin = int((y - h / 2) * h_scale)
        self.xmax = int(self.xmin + w * w_scale)
        self.ymax = int(self.ymin + h * h_scale)
        self.class_id = class_id
        self.confidence = confidence


def IntersectionOverUnion(box_1, box_2):
    width_of_overlap_area = min(box_1.xmax, box_2.xmax) - max(box_1.xmin, box_2.xmin)
    height_of_overlap_area = min(box_1.ymax, box_2.ymax) - max(box_1.ymin, box_2.ymin)
    area_of_overlap = 0.0
    if (width_of_overlap_area < 0.0 or height_of_overlap_area < 0.0):
        area_of_overlap = 0.0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin)
    box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin)
    area_of_union = box_1_area + box_2_area - area_of_overlap
    retval = 0.0
    if area_of_union <= 0.0:
        retval = 0.0
    else:
        retval = (area_of_overlap / area_of_union)
    return retval


def ParseYOLOV3Output(blob, resized_im_h, resized_im_w, original_im_h, original_im_w, threshold, objects):

    out_blob_h = blob.shape[2]
    out_blob_w = blob.shape[3]

    side = out_blob_h
    anchor_offset = 0

    if len(anchors) == 18:   ## YoloV3
        if side == yolo_scale_13:
            anchor_offset = 2 * 6
        elif side == yolo_scale_26:
            anchor_offset = 2 * 3
        elif side == yolo_scale_52:
            anchor_offset = 2 * 0

    elif len(anchors) == 12: ## tiny-YoloV3
        if side == yolo_scale_13:
            anchor_offset = 2 * 3
        elif side == yolo_scale_26:
            anchor_offset = 2 * 0

    else:                    ## ???
        if side == yolo_scale_13:
            anchor_offset = 2 * 6
        elif side == yolo_scale_26:
            anchor_offset = 2 * 3
        elif side == yolo_scale_52:
            anchor_offset = 2 * 0

    side_square = side * side
    output_blob = blob.flatten()

    for i in range(side_square):
        row = int(i / side)
        col = int(i % side)
        for n in range(num):
            obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords)
            box_index = EntryIndex(side, coords, classes, n * side * side + i, 0)
            scale = output_blob[obj_index]
            if (scale < threshold):
                continue
            x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w
            y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h
            height = math.exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1]
            width = math.exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n]
            for j in range(classes):
                class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j)
                prob = scale * output_blob[class_index]
                if prob < threshold:
                    continue
                obj = DetectionObject(x, y, height, width, j, prob, (original_im_h / resized_im_h), (original_im_w / resized_im_w))
                objects.append(obj)
    return objects


def main_IE_infer(objet):
    global fin
    fin=False
    
    camera_width = 320
    camera_height = 240
    fps = ""
    framepos = 0
    frame_count = 0
    vidfps = 0
    skip_frame = 0
    elapsedTime = 0
    new_w = int(camera_width * min(m_input_size/camera_width, m_input_size/camera_height))
    new_h = int(camera_height * min(m_input_size/camera_width, m_input_size/camera_height))

    args = build_argparser().parse_args()
    #model_xml = "lrmodels/tiny-YoloV3/FP32/frozen_tiny_yolo_v3.xml" #<--- CPU
    model_xml = "lrmodels/tiny-YoloV3/FP16/frozen_tiny_yolo_v3.xml" #<--- MYRIAD
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    
#    cap = cv2.VideoCapture(0)
#    cap.set(cv2.CAP_PROP_FPS, 30)
#    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
#    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    camera = PiCamera()
    camera.resolution = (320, 240)
    camera.framerate = 30
    rawCapture = PiRGBArray(camera, size=(320, 240))

    time.sleep(1)

    plugin = IEPlugin(device=args.device)
    if "CPU" in args.device:
        plugin.add_cpu_extension("lib/libcpu_extension.so")
    net = IENetwork(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    exec_net = plugin.load(network=net)

    condition=True
    while (condition):
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            t1 = time.time()
            image=frame.array

    #        ret, image = cap.read()
    #        if not ret:
    #            break
            #retourne l'image
            image = cv2.flip(image, -1)
            
            resized_image = cv2.resize(image, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
            canvas = np.full((m_input_size, m_input_size, 3), 128)
            canvas[(m_input_size-new_h)//2:(m_input_size-new_h)//2 + new_h,(m_input_size-new_w)//2:(m_input_size-new_w)//2 + new_w,  :] = resized_image
            prepimg = canvas
            prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
            prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
            outputs = exec_net.infer(inputs={input_blob: prepimg})

            #output_name = detector/yolo-v3-tiny/Conv_12/BiasAdd/YoloRegion
            #output_name = detector/yolo-v3-tiny/Conv_9/BiasAdd/YoloRegion

            objects = []

            for output in outputs.values():
                objects = ParseYOLOV3Output(output, new_h, new_w, camera_height, camera_width, 0.4, objects)

            # Filtering overlapping boxes
            objlen = len(objects)
            for i in range(objlen):
                if (objects[i].confidence == 0.0):
                    continue
                for j in range(i + 1, objlen):
                    if (IntersectionOverUnion(objects[i], objects[j]) >= 0.4):
                        if objects[i].confidence < objects[j].confidence:
                            objects[i], objects[j] = objects[j], objects[i]
                        objects[j].confidence = 0.0
            
            # Drawing boxes
            label_text=""
            for obj in objects:
                if obj.confidence < 0.2:
                    continue
                label = obj.class_id
                confidence = obj.confidence
                label_text=LABELS[label]
                
                #paramétrage de l'alarme 
                if label_text==objet and obj.confidence>0.3:
                    cv2.imwrite('intrusion.jpg', image)
                    try:
                        dire("Alerte intrusion détectée",125,80)
                        time.sleep(3)
                        pygame.mixer.init()
                        mon_audio=pygame.mixer.Sound("/home/pi/OpenVINO-YoloV3/1462.wav")
                        mon_audio.play()
                    except:
                        print("intrusion détectée")
                    mail()
                    break
                #if confidence >= 0.2:
                label_text = LABELS[label] + " (" + "{:.1f}".format(confidence * 100) + "%)"
                cv2.rectangle(image, (obj.xmin, obj.ymin), (obj.xmax, obj.ymax), box_color, box_thickness)
                cv2.putText(image, label_text, (obj.xmin, obj.ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_text_color, 1)
            
            cv2.putText(image, fps, (camera_width - 170, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
            
            cv2.imshow("Result", image)
            rawCapture.truncate(0)
            
            if cv2.waitKey(1)&0xFF == ord('q'):
                condition=False
                break
            
            elapsedTime = time.time() - t1
            fps = "(Playback) {:.1f} FPS".format(1/elapsedTime)

        ## frame skip, video file only
        #skip_frame = int((vidfps - int(1/elapsedTime)) / int(1/elapsedTime))
        #framepos += skip_frame
    camera.close()
    cv2.destroyAllWindows()
    
    del net
    del exec_net
    del plugin



#--------------------controle du robot et fonction vocale

from tkinter import *
from tkinter import ttk
import serial
import time
import pygame
import os


import subprocess
from functools import partial

ser = serial.Serial('/dev/ttyUSB0', 9600)

#16000 = fréquence utilisée par Raspbian
pygame.mixer.pre_init(16000, -16, 1, 1024)



def dire (sent, pitchou, volou) : # sent = string, pitch =5~255, vol = 0~100
# enonce la phrase au pitch et volume donné
    if type(sent)!=str:
        sent =sent.get()
    pitch=str(pitchou)
    vol=str(volou)
    cmdd="<pitch level='"+pitch+"'> "+"<volume level='"+vol+"'> "
    cmdd='"'+cmdd+sent+'"'
    cmd='pico2wave -l fr-FR -w fra.wav '+cmdd
    print(cmd)
    os.system(cmd)
    pygame.mixer.init()
    mon_audio=pygame.mixer.Sound("/home/pi/OpenVINO-YoloV3/fra.wav")
    mon_audio.play()
    #time.sleep(5)
    #mon_audio.stop()

def direction (commande):
    for i in range (0,2):
        ser.write(commande.encode('ascii'))
    commande="s"
    ser.write(commande.encode('ascii'))

def dir_cam (commande):
    ser.write(commande.encode('ascii'))
    commande="s"
    ser.write(commande.encode('ascii'))
    
def dep_haut(event):
    commande="a"
    direction(commande)
    
def dep_bas(event):
    commande="r"
    direction(commande)
    
def dep_gauche(event):
    commande="g"
    direction(commande)
    
def dep_droite(event):
    commande="d"
    direction(commande)

def h_droite(event):
    commande="j"
    dir_cam(commande)
    
def h_gauche(event):
    commande="h"
    dir_cam(commande)
    
def v_haut(event):
    commande="f"
    dir_cam(commande)
    
def v_bas(event):
    commande="v"
    dir_cam(commande)

def yolo(event):
    global pi_cam
    show(False,pi_cam)
    try:
        dire ("lancement du programme de détection d'intrusion",125,80)
    except:
        print("lancement du programme de détection d'intrusion")
    objet="person"
    main_IE_infer(objet)
    try:
        dire("fin du programme de détection d'intrusion",125,80)
    except:
        print("fin du programme de détection d'intrusion")
    pi_cam=show(True,pi_cam)
    
def detect(objet):
    if type(objet)!=str:
        objet = objet.get()
    print("contenu de la variable objet :",objet)
    global pi_cam
    show(False,pi_cam)
    try:
        dire ("lancement du programme de détection d'intrusion",125,80)
    except:
        print("lancement du programme de détection d'intrusion")
    objet=objet
    main_IE_infer(objet)
    try:
        dire("fin du programme de détection d'intrusion",125,80)
    except:
        print("fin du programme de détection d'intrusion")
    pi_cam=show(True,pi_cam)


def end_fullscreen(event):
    root.attributes("-fullscreen", False)
    
"Paramètrage Joystick"

class Find_Joystick:
    global valeur_pad
    def __init__(self, root):
        self.root = root
        ## initialize pygame and joystick
        pygame.init()
        if(pygame.joystick.get_count() < 1):
            # no joysticks found
            print ("Please connect a joystick.")
            self.quit()
        else:
            # create a new joystick object from
            # ---the first joystick in the list of joysticks
            Joy0 = pygame.joystick.Joystick(0)
            # tell pygame to record joystick events
            Joy0.init()

        ## bind the event I'm defining to a callback function
        self.root.bind("<<JoyFoo>>", self.my_event_callback)

        ## start looking for events
        self.root.after(0, self.find_events)

    def find_events(self):
        #print(valeur_pad)
        pause=0
        if valeur_pad==(0,1):
            commande="a"
            direction(commande)
            time.sleep(pause)

        if valeur_pad==(0,-1):
            commande="r"
            direction(commande)
            time.sleep(pause)

        if valeur_pad==(1,0):
            commande="d"
            direction(commande)
            time.sleep(pause)

        if valeur_pad==(-1,0):
            commande="g"
            direction(commande)
            time.sleep(pause)
            
        ## check everything in the queue of pygame events
        events = pygame.event.get()
        for event in events:
            try:
                print(event.type)
                print(event.button)
                print(event.value)
            except:
                print("Erreur")
            finally:
            # event type for pressing any of the joystick buttons down
                if event.type == pygame.JOYBUTTONDOWN:
                    if event.button==7:
                        commande="a"
                        direction(commande)
                # generate the event I've defined
                    #self.root.event_generate("<<JoyFoo>>")
                if event.type == pygame.JOYHATMOTION:
                # generate the event I've defined
                    self.root.event_generate("<<JoyFoo>>")
                                
        ## return to check for more events in a moment
        self.root.after(20, self.find_events)

    def my_event_callback(self, event):
        print ("JPad button press")
        global valeur_pad
        valeur_pad=pygame.joystick.Joystick(0).get_hat(0)

    ## quit out of everything
    def quit(self):
        import sys
        sys.exit()

#---------------------interface graphique

root = Tk()
root.geometry('650x300+20+10')
#root.configure(bg="white")
#root.attributes("-fullscreen", True)
root.title("Curiosity")


label_1 = Label(root, text=" ---- Curiosity --- ", font="Verdana 26 bold", fg="#000")
label_1.grid(row=0, column=0)

#----synthétiseur
cadre = Frame(root, bg='blue', width=300, height=200, borderwidth=2)
cadre.grid(row=2,column=0)

text = StringVar()
#text = ttk.Combobox(root, textvariable=text)
text = ttk.Combobox(cadre, textvariable=text)
text['values'] = ('Coucou', 'Coucou Louise la crapule', 'Coucou Léonard la touffe et Louise la crapule', 'défrites défritedéfritte')
text.grid(row=2, column=0)
#text.pack(side="top", fill=X)

label = Label(root, text='Synthétiseur vocal')

button = Button(root, text='Dire', 
                command=partial(dire, text, 125,80))
                
label.grid(column=0, row=1)
#entry_name.grid(column=0, row=2)
button.grid(column=0, row=3)

#----deplacement du rover
avancer="a"
reculer="r"
gauche="g"
droite="d"

#bouton déplacement

bouton_avancer = Button(root, text="Avancer", command=partial(direction, avancer))
bouton_reculer = Button(root, text="Reculer", command=partial(direction, reculer))
#bouton_stop = Button(fenetre, text="Stop", command=stop)
bouton_droite = Button(root, text="Droite", command=partial(direction, droite))
bouton_gauche = Button(root, text="Gauche", command=partial(direction, gauche))

#quitter
bouton_quitter= Button(root, text="Quitter", command=quitter)

#On les affiches
bouton_avancer.grid(row =0, column =2)
bouton_reculer.grid(row =2, column =2)
#bouton_stop.grid(row =1, column =2)
bouton_droite.grid(row =1, column =3)
bouton_gauche.grid(row =1, column =1)

bouton_quitter.grid(row=9, column=0)


#deplacement caméra
droite="j"
gauche="h"
haut="f"
bas="v"

label_cam = Label(root, text='Caméra')
label_cam.grid(row=5, column=2)

cam_droite=Button(root, text="Droite", command=partial(dir_cam, droite))
cam_gauche=Button(root, text="Gauche", command=partial(dir_cam, gauche))
cam_haut=Button(root, text="Haut", command=partial(dir_cam, haut))
cam_bas=Button(root, text="Bas", command=partial(dir_cam, bas))

cam_droite.grid(row=5, column=3)
cam_gauche.grid(row=5, column=1)
cam_haut.grid(row=4, column=2)
cam_bas.grid(row=6, column=2)

#objets détectables
objet = StringVar()
objet = ttk.Combobox(root, textvariable=objet)
objet['values'] = ('person', 'dog', 'cat', 'car')
objet.grid(row=5, column=0)


#détection objets
detection=Button(root, text="Détection", command=partial(detect,objet))
detection.grid(row=6, column=0)


#label
label_enregistrement = Label(root, text='Enregistrement')
label_enregistrement.grid(row=7, column=1)

label_zoom = Label(root, text='Zoom')
label_zoom.grid(row=7, column=3)

#photo
app_photo=Button(root, text="Photo", command=photo)
app_photo.grid(row=8, column=1)

#video
app_video=Button(root, text="Video", command=video)
app_video.grid(row=9, column=1)

#zoom
app_zoom_in=Button(root, text="Zoom  _in", command=zoom_in)
app_zoom_out=Button(root, text="Zoom out", command=zoom_out)

app_zoom_in.grid(row=8, column=3)
app_zoom_out.grid(row=9, column=3)

#-----------------------------début programme-------------
photo=False
video=False
zoom=False
photo_prise=0
video_prise=0
pi_cam=show_webcam()
quitter=False

print("lancement fonction show")
print("nom du thread", pi_cam.name)
pi_cam=show(True,pi_cam)
print("nom du thread après fonction show", pi_cam.name)

try :
    valeur_pad=(0,0)
    app = Find_Joystick(root)

except:
    print("erreur")
#finally :

#root.bind("<Up>", haut)
root.bind("<Up>", dep_haut)
root.bind("<Down>", dep_bas)
root.bind("<Left>", dep_gauche)
root.bind("<Right>", dep_droite)
root.bind("j", h_droite)
root.bind("h", h_gauche)
root.bind("f", v_haut)
root.bind("v", v_bas)
root.bind("#", yolo)
#root.bind("z", app_video)
root.bind("<Escape>", end_fullscreen)

    

root.mainloop()
