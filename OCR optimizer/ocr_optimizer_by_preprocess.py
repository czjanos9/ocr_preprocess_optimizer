import os
import cv2
from cv2 import IMREAD_UNCHANGED
import numpy as np
import pytesseract as tess
import pickle
import tkinter

from tkinter import filedialog
from matplotlib.pyplot import clf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

tess.pytesseract.tesseract_cmd =r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#####--- Image related functions ---#####
#########################################
###--- Do nothing function ---###
def empty():
    pass

###--- STACK IMAGES ---###
def stackImages(scale,imgArray):
    
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)

        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])

        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:

                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:

                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)

        hor= np.hstack(imgArray)
        ver = hor

    return ver 

###--- Get contours of image function ---###
#Inputs:    image,
#           min desired size of area          
#
#Output:    Area (bigger than areaSize and have 4 edge)
#           image with drawed area
def getContours(img, imgContour, areaSize=100000):
    biggest = np.array([])
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > areaSize:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True) 
            if len(approx) == 4:
                biggest = approx
                         
            cv2.drawContours(imgContour, biggest, -1, (255,0,0), 200)
            return biggest

###--- Image preprocess function ---###
#Inputs:    image,
#           Gaussian blur Density, 
#           Dialate No. of iterations,
#           Erode No. of iterations
#
#Output:    Preprocessed image
def preProcessing(img,blurDens,cannyDens,dilateIter,erodeIter):
    imgBlur = cv2.GaussianBlur(img, (5,5),blurDens)
    imgCanny = cv2.Canny(imgBlur,cannyDens,cannyDens)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=dilateIter)
    imgThres = cv2.erode(imgDial,kernel,iterations=erodeIter)
    
    return imgThres

###--- Set perspetcive ---###
#Inputs:    image,
#           coords of area
#           desired window width   
#           desire window height
#
#Output:    Warped image
def reorder(myPoints):
    #This function reorder the coords for desired perspective
    
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    
    myPointsNew[0] = myPoints[np.argmin(add)] #smallest value
    myPointsNew[3] = myPoints[np.argmax(add)] #largest value
    diff = np.diff(myPoints,1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def getWarp(img,biggest,windowWidth,windowsHeight):
    ratio = windowsHeight / windowWidth
    objectTop = abs(np.sqrt(np.square(biggest[0][0][0] - biggest[1][0][0]) + np.square(biggest[0][0][1] - biggest[1][0][1]))) 
    objectBottom = abs(np.sqrt(np.square(biggest[2][0][0] - biggest[1][0][0]) + np.square(biggest[0][0][1] - biggest[1][0][1]))) 
   
    if objectTop > objectBottom:
        objectWidth = objectTop
    else:
        objectWidth = objectBottom
    
    newWidth = np.int32(round(objectWidth * ratio))
    newHeight = np.int32(round(objectWidth ))
    
    biggest = reorder(biggest)
    points1 = np.float32(biggest)
    points2 = np.float32([[0,0],[newHeight,0],[0,newWidth],[newHeight,newWidth]])
    matrix = cv2.getPerspectiveTransform(points1,points2)
    imgOutput = cv2.warpPerspective(img,matrix,(newHeight,newWidth)) 
    return imgOutput

###--- OCR Call ---###
#Input: Image,
#       labelling=True or False
#
#Outputs:   image with word boxes,
#           exctracted text
#
#Draw a box on picture, where an actual word is found
#If labbeling set to true, then its write the charaters under the boxes
def callOCR(image):
    extracted_text = ""
    boxes = tess.image_to_data(image, lang='hun')
    #print(boxes)
    imgReturn = image.copy()
    try:
        imgReturn = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    except cv2.error as error:
        print('The given image has already 3 channel instead of 1')
        print('Passing...')
    for x,b in enumerate(boxes.splitlines()):
        if x != 0:
            b = b.split() #Its make a list, so we can access it invidualy
            #Where the list size is 11 there is an actual word, we sort the rest out
            #print(b)
            if len(b) == 12:
                x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                cv2.rectangle(imgReturn,(x,y),(w+x,h+y),(0,0,255),2)
                extracted_text = extracted_text + "\n" + str(b[11])
              
    return imgReturn,extracted_text

""" def preProcessStepByStep(img,blurVal,cannyVal,dilateIter,erodeIter, size=0.025):
    wImg,hImg,_ = img.shape
    imgThres = preProcessing(img,blurVal,cannyVal,dilateIter,erodeIter)
    imgContour = img.copy()
    biggest = getContours(imgThres, imgContour)
    imgResult = img.copy()
    if biggest is not None:
        if biggest != []:
            imgWarped = getWarp(img,biggest,wImg,hImg)
        else:
            imgWarped = img
            imgContour = img.copy()   
    else:
        imgWarped =  img

    imgGray = cv2.cvtColor(imgWarped,cv2.COLOR_RGB2GRAY)   
    th, imgBinarized = cv2.threshold(imgGray,254,254,cv2.THRESH_OTSU)
    if cv2.waitKey(10) & 0xFF == ord('t'):
        imgResult,extracted_text = callOCR(imgBinarized)
        print(extracted_text)
    
    stackedImg = stackImages(size,([img,imgThres,imgContour],[imgWarped,imgBinarized,imgResult]))
    return stackedImg, imgBinarized
    cv2.imshow('Steps',stackedImg)
    if cv2.waitKey(10) & 0xFF == ord('s'):
            cv2.imwrite("img.jpg",img)
            cv2.imwrite("imgThres.jpg",imgThres)
            cv2.imwrite("imgContour.jpg",imgContour)
            cv2.imwrite("imgWarped.jpg",imgWarped)
            cv2.imwrite("imgBinarized.jpg",imgBinarized)
            cv2.imwrite("imgResult.jpg",imgResult)
            print("Save Done!")  
        
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break """
def doWarp(img, imgThres):
    imgContour = img.copy()
    wImg,hImg,_ = img.shape
    biggest = getContours(imgThres, imgContour)
    if biggest is not None:
        if biggest != []:
            imgWarped = getWarp(img,biggest,wImg,hImg)
        else:
            imgWarped = img
            imgContour = img.copy()   
    else:
        imgWarped =  img
    return imgWarped, imgContour

###--- GET DATA FUNCTIONS---###
def get_data(filename):
    with open(filename, "r", encoding='utf-8') as f:
        data = f.read()[0:424431]
    data = data.lower()
    return (data)

def get_all_folder(path_to_folder):
    dirlist = os.listdir(path_to_folder)
    return dirlist

def get_all_file(path_to_folder):
    dirlist = os.listdir(path_to_folder)
    file_paths = []
    image_paths = []
    #print(dirlist)
    for file in dirlist:
        if os.path.isfile(path_to_folder + '/' + file) and (file.find(".txt") != -1):
            file_paths.append(path_to_folder + '/' + file)
        if os.path.isfile(path_to_folder + '/' + file) and ((file.find(".jpg") or file.find(".png") != -1)):
            image_paths.append(path_to_folder + '/' + file)

    return (file_paths, image_paths)

def readDataSet(label_folder,ocred_texts_folder):

    label_paths, label_image_paths = get_all_file(label_folder)
    
    folders = get_all_folder(ocred_texts_folder) #Meg keresi az összes mappát
    text_paths = []
    image_paths = []
    for folder in folders:
        files, images =  get_all_file(ocred_texts_folder + '/' + folder)
        text_paths.append(files)
        image_paths.append(images)
    #label_folder és file_paths tartalmazza az eléris útvonalakat
    return label_paths,text_paths

###--- TDATA CLASS ---####
class TData:
    def __init__(self):
        self.size = -1
        self.label_step = -1
        self.ocred_step = -1
        self.unique_names = []
        self.label_names = []
        self.label_paths = []
        self.label_texts = []
        self.ocred_names = []
        self.ocred_paths = []
        self.ocred_texts = []
        self.ratios = []           
        self.blurs = []            
        self.cannies = []            
        self.dilates = []
        self.erodes = []

    def add(self, label_path, ocred_path):

        temp = ocred_path.rsplit('/', 1)
        if '(' in temp[0]:
            temp[0] = temp[0][0:temp[0].index('(')]
        temp_parameters = temp[0].split('-')
        temp_label_name = label_path.split('/')
        temp_ocred_name = ocred_path.split('/')

        if not hasattr(self, 'size'):
            self.size = -1
            self.label_names = []
            self.label_paths = []
            self.label_texts = []
            self.ocred_names = []
            self.ocred_paths = []
            self.ocred_texts = []           
            self.blurs = []            
            self.cannies = []            
            self.dilates = []
            self.erodes = []
            
        self.size = self.size + 1
        self.label_names.append(temp_label_name[len(temp_label_name)-1])        
        self.label_paths.append(label_path)
        self.label_texts.append(get_data(label_path))
        self.ocred_names.append(temp_ocred_name[len(temp_ocred_name)-1])
        self.ocred_paths.append(ocred_path)
        self.ocred_texts.append(get_data(ocred_path))
        self.blurs.append(temp_parameters[len(temp_parameters)-4])
        self.cannies.append(temp_parameters[len(temp_parameters)-3])
        self.dilates.append(temp_parameters[len(temp_parameters)-2])
        self.erodes.append(temp_parameters[len(temp_parameters)-1])
        if (self.size > 1 ):
            if (self.label_step == -1 and self.label_names[self.size] != self.label_names[self.size-1]):
                self.label_step = self.size

    def toString(self, index=0):
        return (F'[{index}]----- {self.label_names[index]}|{self.ocred_names[index]} -----\nLabel Path:{self.label_paths[index]}\nText Path:{self.ocred_paths[index]}\nRatio:{self.ratios[index]}\nParameters: {self.blurs[index]}, {self.cannies[index]}, {self.dilates[index]}, {self.erodes[index]}\n')

    def print(self, index=0):
        print(F'|----- {self.label_names[index]}|{self.ocred_names[index]} -----|\nLabel Path:{self.label_paths[index]}\nText Path:{self.ocred_paths[index]}\nRatio:{self.ratios[index]}\nParameters: {self.blurs[index]}, {self.cannies[index]}, {self.dilates[index]}, {self.erodes[index]}\n--------------------')

    def setStep(self, step):
        self.ocred_step = step

##############################################################################################################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################################################################################################
###--- Main(Manual) ---###
def manualOCR():
    ###--- UI for manual settings ---###
    cv2.namedWindow("Settings")
    cv2.resizeWindow("Settings", 800,365)
    #cv2.createTrackbar("Threshold","Settings", 128, 255, empty)
    #cv2.createTrackbar("MaxValue","Settings", 255, 255, empty)
    cv2.createTrackbar("Blur Density","Settings", 200, 500, empty)
    cv2.createTrackbar("Canny Density","Settings", 100, 500, empty)
    cv2.createTrackbar("Dilate Iteration","Settings", 6, 30, empty)
    cv2.createTrackbar("Erode Iteration","Settings", 1, 10, empty)
    cv2.createTrackbar("Output image size","Settings", 1, 25, empty)
    cv2.setTrackbarPos("Output image size","Settings", 1)

    #initiate tinker and hide window
    print('Choose a file.') 
    main_win = tkinter.Tk() 
    main_win.withdraw()

    main_win.overrideredirect(True)
    main_win.geometry('0x0+0+0')

    main_win.deiconify()
    main_win.lift()
    main_win.focus_force()


    while not hasattr(main_win, 'sourceFile'):
        #open file selector 
        main_win.sourceFile = filedialog.askopenfilename(parent=main_win, initialdir= "/",
        title='Válassz egy képfálj-t!')

        #close window after selection 
        main_win.destroy()

        #print path 
        print(main_win.sourceFile )

    path = main_win.sourceFile
    try:
        img = cv2.imread(path)
        if (".png" in path.lower()):
            img = cv2.imread(path, IMREAD_UNCHANGED)
            jpg_path = path[:-3] + 'jpg'
            print(jpg_path)
            cv2.imwrite(jpg_path, img)
            img = cv2.imread(jpg_path)
        imgResult = img.copy()

        conf = r' --oem 3 -- psm 6 outputbase'
                #its detects only digits
                #--oem -> Represents the engine mode DOCS->
                #--psm -> Page segmentation mode DOCS->

        hImg, wImg,_ = img.shape

        while True:
            #thresh = cv2.getTrackbarPos("Threshold","Settings")
            blurVal = cv2.getTrackbarPos("Blur Density","Settings")
            cannyVal = cv2.getTrackbarPos("Canny Density","Settings")
            dilateIter = cv2.getTrackbarPos("Dilate Iteration","Settings")
            erodeIter = cv2.getTrackbarPos("Erode Iteration","Settings") 
            size0 = cv2.getTrackbarPos("Output image size","Settings")
            if(size0 >= 1): size = size0*0.025
            imgThres = preProcessing(img,blurVal,cannyVal,dilateIter,erodeIter)
            #imgContour = img.copy()
            imgWarped, imgContour = doWarp(img, imgThres)

            imgGray = cv2.cvtColor(imgWarped,cv2.COLOR_RGB2GRAY)   
            th, imgBinarized = cv2.threshold(imgGray,254,254,cv2.THRESH_OTSU)
            if cv2.waitKey(10) & 0xFF == ord('t'):
                imgResult,extracted_text = callOCR(imgBinarized)
                print(extracted_text)

            stackedImg = stackImages(size,([img,imgThres,imgContour],[imgWarped,imgBinarized,imgResult]))
            #stackedImg, imgBinarized = preProcessStepByStep(img,blurVal,cannyVal, dilateIter, erodeIter, size0)
            ###RAM PROBLEM
            cv2.imshow('Steps',stackedImg)
            if cv2.waitKey(10) & 0xFF == ord('t'):
                imgResult,extracted_text = callOCR(imgBinarized)
                print(extracted_text)
            if cv2.waitKey(10) & 0xFF == ord('s'):
                    print('Saving...')
                    cv2.imwrite("img.jpg",img)
                    cv2.imwrite("imgThres.jpg",imgThres)
                    cv2.imwrite("imgContour.jpg",imgContour)
                    cv2.imwrite("imgWarped.jpg",imgWarped)
                    cv2.imwrite("imgBinarized.jpg",imgBinarized)
                    cv2.imwrite("imgResult.jpg",imgResult)
                    print("Save Done!")  
                
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break 
    except cv2.error as error:
        print(error)
        print("Ügyeljen arra, hogy az eléris útvonalban ne legyen speciális és ékezetes karakter!")

###--- Main(createdata) ---###
def createData(path_to_labels="pictures_labels/", withImage=False):
    number_of_files = 0
    dirlist = os.listdir(path_to_labels)
    for file in dirlist:
        if ".txt" in file:
            number_of_files = number_of_files + 1
    #print(number_of_files)

    for step in range(1,number_of_files):
        imagePath = F'{path_to_labels}/{step}.jpg'
        print(F'step:{step}/{number_of_files}')

        tempPath = 'temp/'

        image_file = imagePath
        img = cv2.imread(image_file)    
        
        imgContour = img.copy()
        hImg,wImg,_ = img.shape

        next_threshold = 250
        next_maxvalue = 250
        next_blur = np.round(np.linspace(1,50,3))
        next_canny = np.round(np.linspace(1,100,10))
        next_dilate = np.round(np.linspace(1,5,4))
        next_erode = np.round(np.linspace(0,5,5))
        next_area = 100000
        next_width = 600
        next_height = 800
        state = 0
        all_state = len(next_blur) * len(next_canny) * len(next_dilate) * len(next_erode)

        print(F'blur:{len(next_blur)} canny:{len(next_canny)} dilate:{len(next_dilate)} erode:{len(next_erode)}')
        for blur_iter in next_blur.astype(int):    
            for canny_iter in next_canny.astype(int):
                for dilate_iter in next_dilate.astype(int):
                    for erode_iter in next_erode.astype(int):
                        i = 0
                        path = F"test/test-{blur_iter}-{canny_iter}-{dilate_iter}-{erode_iter}"
                        if not os.path.exists(path):
                            os.makedirs(path)
                        else:
                            path = F"test/test-{blur_iter}-{canny_iter}-{dilate_iter}-{erode_iter}({i})"
                            while  os.path.exists(path):
                                i = i + 1
                                path = F"test/test-{blur_iter}-{canny_iter}-{dilate_iter}-{erode_iter}({i})"
                            os.makedirs(path)

                        imgThres = preProcessing(img,blur_iter,canny_iter,dilate_iter,erode_iter)
                        biggest = getContours(imgThres, next_area)
                        if biggest is not None:
                            if biggest != []:
                                imgWarped = getWarp(img,biggest,wImg,hImg)
                            else:
                                imgWarped = img
                                imgContour = img.copy() 
                        else:
                            imgWarped =  img

                        imgGray = cv2.cvtColor(imgWarped,cv2.COLOR_RGB2GRAY)   
                        th, imgBinarized = cv2.threshold(imgGray,next_threshold,next_maxvalue,cv2.THRESH_OTSU)

                        #binarization on end:
                        th2, image_with_border = cv2.threshold(imgGray,next_threshold,next_maxvalue,cv2.THRESH_OTSU)
                        imgResult,extracted_text = callOCR(imgBinarized)

                        if withImage: cv2.imwrite(F"{path}/imgResultA-{blur_iter}-{canny_iter}-{dilate_iter}-{erode_iter}.jpg",imgResult)
                        f = open(F"{path}/currResultA-{blur_iter}-{canny_iter}-{dilate_iter}-{erode_iter}.txt", "w+", encoding="utf-8")
                        f.write(str(extracted_text))
                        f.close
                        state = state + 1
                        print(F"Process: {state}/{all_state}")
                        print("Save Done!")
    ##Return to menu
    showMenu()

def createTDataFile(filename = "tdata.pk"):
    try:
        #choice = int(input("Choose option:\n[0]Create TData\n[1]Load TData\n->"))       
        #if choice == 0:
            print('Default label path: ../pictures_labels/')
            print('Default generatedData path: ../trainData/')

            label_folder = 'pictures_labels/'
            ocred_texts_folder = 'trainData/'
            label_paths,text_paths = readDataSet(label_folder,ocred_texts_folder)

            tdata = TData()
            ocred = []

            for file in text_paths:
                index = str(file).find("(")
                index = int(index)
                
                if (index == -1) and len(ocred) < 1:
                    ocred.append([])
                    ocred[0].append(str(file))
                elif (index == -1) and len(ocred) >= 1:
                    ocred[0].append(str(file))
                
                if (index != -1):
                    index_end = str(file).find(")")
                    number = str(file[0][index-1:index_end-2])
                    number = int(number)
                    if len(ocred) <= number + 1: #offset
                        for i in range(len(ocred),number+2):
                            ocred.append([]) #üres
                    
                    ocred[number+1].append(str(file))

            tdata.setStep(len(ocred[0])-1)

            for i in range(0, len(label_paths)-1):
                    for j in range(0, len(ocred)-1):
                        print(f'{i+1}/{len(label_paths)-1}, {j+1}/{len(ocred)-1}')
                        for k in range(0, len(ocred[j])-1):
                            temp = ""
                            temp = temp + str(ocred[j][k][2:len(ocred[j][k])-2])
                            tdata.add(label_paths[i],temp)



            ###--- Save TData ---###
            print(F'Saving data to pickle file to {filename}...')
            with open(filename, 'wb') as file:
                pickle.dump(tdata, file)
            print('Save done!')
            #data_dict.dump(tdata)
                
        #else:
        #    ###--- Load TData ---###
        #    print(F'Loading data from {filename}...')
        #    with open(filename, 'rb') as file:
        #        tdata = pickle.load(file)
        #        print(f'{tdata.size} label-ocred text pairs are loaded')
    except ValueError as error:
        print(error)



def doPredict(filename='tdata.pk'):
    ###--- Load TData ---###
    print(F'Loading data from {filename}...')
    with open(filename, 'rb') as file:
        tdata = pickle.load(file)
        print(f'{tdata.size} label-ocred text pairs are loaded')
    print('Load complete.')

    #initiate tinker and hide window
    print('Choose a file.')  
    main_win = tkinter.Tk() 
    main_win.withdraw()

    main_win.overrideredirect(True)
    main_win.geometry('0x0+0+0')

    main_win.deiconify()
    main_win.lift()
    main_win.focus_force()

    while not hasattr(main_win, 'sourceFile'):
        #open file selector 
        main_win.sourceFile = filedialog.askopenfilename(parent=main_win, initialdir= "/",
        title='Válassz egy képfáljt melyen a felismerést szeretnéd elvégezni!')

        #close window after selection 
        main_win.destroy()

        #print path 
        print(main_win.sourceFile )
    path = main_win.sourceFile
    corpus_label = []
    corpus_sample = []

    for i in range(0,tdata.size,tdata.label_step):
        corpus_label.append(tdata.label_texts[i])

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus_label)
    array_x = X.toarray()
    #109016
    results = []
    scores = []
    iter = -1
    for n in range(0,tdata.size,tdata.label_step):
        iter = iter + 1
        corpus_sample = []
        if iter == 0:
            off_set = 0
        else:
            off_set = tdata.ocred_step*iter #599#

        for i in range(0,tdata.ocred_step-1,1):
            corpus_sample.append(tdata.ocred_texts[i+off_set])

        X_sample = vectorizer.transform(corpus_sample)
        array_sample = X_sample.toarray()

        best_score = -1
        best_index = -1
        best_count = 0
        for k in range(0,len(array_sample)):
            count = 0
            score = 0
            for j in range(0,len(array_sample[k])):
                if(array_sample[k][j] != 0):
                    count = count + 1
                    score = score + array_sample[k][j]
                    if len(corpus_label[iter]) < score:
                        score = score - (score - len(corpus_label[iter]))
            if best_score < score:
                best_score = score
                best_index = k
                best_count = count

        result = best_index+off_set
        results.append(result)
        scores.append(best_score)
        print(F'[{iter}]The Best Score:{best_score} at: {best_index} {best_count}/{len(corpus_sample[best_index])}')
        f = open(F"bag_of_words_results/result-{iter}.txt", "w+", encoding="utf-8")
        f.write(F'{tdata.ocred_paths[result]}\nBlur:{tdata.blurs[result]},Canny:{tdata.cannies[result]},dilate:{tdata.dilates[result]},Erode:{tdata.erodes[result]},\n{corpus_sample[best_index]}')
        f.close()

    print(F'Results:\n {results}')
    print('####################')
    print(F'Scores:\n{scores}')
    print('\n')

    categories = []
    target = []
    for i in range(1, len(results)+1): #len(results)-1
        categories.append(i)
        target.append(i)
    #target = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    count_vect = CountVectorizer()

    X_train_counts = count_vect.fit_transform(corpus_label) ##full corpus
    print(F'X_train shape:{X_train_counts.shape}')

    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    print(F'X_train_tf shape:{X_train_tf.shape}')

    clf = MultinomialNB().fit(X_train_tf,target)

    #docs_new = ['Pécsi tudományegyetem Intel processzrok programozása Gimesi László']
    #docs_new = ['A RISC processzor minden egyszerű parancsot közvetlenül végre tud hajtani, így nincsenek mikroutasítások, nincs interpreter, és nincs szükség a mikroprogram-memóriára sem. Ebből']

    ###--- Run tesseract without addotional preprocess
    print(path)
    img = cv2.imread(path)
    print('Call default ocr...')
    imgGray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)   
    th, imgBinarized = cv2.threshold(imgGray,254,254,cv2.THRESH_OTSU)
    o_result_img, o_result_text = callOCR(imgBinarized)
    o_corpus = []
    o_corpus.append(o_result_text.replace('\n', ' '))
    #o_corpus.append(o_result_text)
    print(len(o_corpus))
    print(o_corpus)
    if len(o_result_text) != 0:
        
        X_new_counts = count_vect.transform(o_corpus)
        X_new_tf = tf_transformer.transform(X_new_counts)
        print(X_new_tf.shape)

        #Predict which label text fit to new data
        predicted = clf.predict(X_new_tf)

        for doc, category in zip(o_result_text, predicted): 
            print(f"{doc} => {category}")

        params = []
        params.append(int(tdata.blurs[results[category-1]]))
        params.append(int(tdata.cannies[results[category-1]]))
        params.append(int(tdata.dilates[results[category-1]]))
        params.append(int(tdata.erodes[results[category-1]]))
        print(F'Params to call:{tdata.blurs[results[category-1]]},{tdata.cannies[results[category-1]]},{tdata.dilates[results[category-1]]},{tdata.erodes[results[category-1]]}')

        #Do preprocessing with these parameters
        n_imgThres = preProcessing(img,params[0],params[1],params[2],params[3])
        imgWarped, imgContour = doWarp(img, n_imgThres)
        n_imgGray = cv2.cvtColor(imgWarped,cv2.COLOR_RGB2GRAY)   
        th, n_imgBinarized = cv2.threshold(n_imgGray,200,254,cv2.THRESH_OTSU)
        n_result_img, n_result_text = callOCR(n_imgBinarized)

        print(F"length of non preprocessed text:{len(o_result_text)}\nlength of preprocessed text:{len(n_result_text)}")
        f = open(F"non_preprocessed.txt", "w+", encoding="utf-8")
        f.write(o_result_text)
        f.close
        f = open('preprocessed.txt', 'w+', encoding='utf-8')
        f.write(n_result_text)
        f.close

    else:
        print('The default ocr gave no results, trying the best paramtere settings...')
        score = -1
        index = -1
        for i in range(0,len(scores)-1):
            if score < scores[i]:
                score = scores[i]
                index = i
        print(F'Do ocr with the highest score parameter: {score}')
        print(F'Params to call:{tdata.blurs[results[index]]},{tdata.cannies[results[index]]},{tdata.dilates[results[index]]},{tdata.erodes[results[index]]}')
        

        
        n_imgThres = preProcessing(img,int(tdata.blurs[results[index]]),int(tdata.cannies[results[index]]),int(tdata.dilates[results[index]]),int(tdata.erodes[results[index]]))
        #n_imgThres = preProcessing(img,1,1,1,0)
        imgWarped, imgContour = doWarp(img, n_imgThres)
        n_imgGray = cv2.cvtColor(imgWarped,cv2.COLOR_RGB2GRAY)   
        th, n_imgBinarized = cv2.threshold(n_imgGray,254,254,cv2.THRESH_OTSU)
        n_result_img, n_result_text = callOCR(n_imgBinarized)

        print(F"length of non preprocessed text:{len(o_result_text)}\nlength of preprocessed text:{len(n_result_text)}")
        f = open(F"non_preprocessed.txt", "w+", encoding="utf-8")
        f.write(str(o_result_text))
        f.close
        f = open('preprocessed.txt', 'w+', encoding='utf-8')
        f.write(str(n_result_text))

###--- Show menu ---###
def showMenu():
    while True:
        choice = int(input("Choose option:\n[0]Test manual\n[1]Test create dataset (label-text required)\n[2]Test create TData (dataset required)\n[3]Test predict parameters (tdata.pk required)\n[4]Quit\n->"))
        if(choice == 0):manualOCR()
        elif(choice == 1):createData()
        elif(choice == 2):createTDataFile()
        elif(choice == 3):doPredict()
        elif(choice == 4):break
        else: print('\n->')


###--- MAIN ---###
showMenu()
    





