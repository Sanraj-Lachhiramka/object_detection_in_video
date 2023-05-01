import cv2 as cv
#import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
 
yolo=cv.dnn.readNet("yolov3.weights","yolov3.cfg.txt")

#coco.names contain list of names that our algo can detect

classes=[]
with open("coco.names",'r') as f:
    classes=f.read() .splitlines() #classes-contins the  name of different object it can detect

#for image
cap=cv.VideoCapture(0)

while True:
    ret,img=cap.read()
    cv.imshow("img1",img)
    blob=cv.dnn.blobFromImage(img,1/255,(320,320),(0,0,0),swapRB=True,crop=False)#1/255 is for regularization #resize the img #swap is true because it is going to read image in bgr so it must swap r and b
    i=blob[0].reshape(320,320,3)
    plt.imshow(i)

    yolo.setInput(blob) # we have assigned blob as an input image

    output_layes_name=yolo.getUnconnectedOutLayersNames()
    layeroutput=yolo.forward(output_layes_name)

    boxs=[]
    confidences=[]
    class_ids=[]

    width=img.shape[1]
    height=img.shape[0]

    for output in layeroutput:
        for detection in output:
            score=detection[5:]
            class_id=np.argmax(score)
            confidence=score[class_id]
            if confidence>0.7:
                center_x=int(detection[0]*width)
                center_y=int(detection[1]*height)
                w=int(detection[2]*width)
                h=int(detection[3]*height)

                x=int(center_x-w/2)
                y=int(center_y-h/2)

                boxs.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
            #print(detection)

    indexes=cv.dnn.NMSBoxes(boxs,confidences,0.5,0.4)

    font=cv.FONT_HERSHEY_PLAIN

    COLORS=np.random.uniform(0,255,size=(len(boxs),3))

    if len(indexes) > 0:
        for i in indexes.flatten():
            x,y,w,h=boxs[i]
            label=str(classes[class_ids[i]])
            confi=str(round(confidences[i],2))
            color=COLORS[i]

            font=cv.FONT_HERSHEY_COMPLEX

            cv.rectangle(img,(x,y),(x+w,y+h),color,20)
            cv.putText(img,label+" "+confi,(x,y+20),font,1,(255,255,255),3,cv.LINE_AA)

    plt.imshow(img)

    cv.imshow("img",img)
    if cv.waitKey(1)==13:
        break
    


cv.destroyAllWindows()
cap.release()