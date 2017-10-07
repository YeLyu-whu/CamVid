from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os
import numpy as np
from matplotlib import pyplot as plt 
from PIL import Image


def isImageFile(filePath):
    basename_ext = os.path.splitext(filePath)
    ext= basename_ext[1]
    ext = ext.lower()
    if ext=='.bmp': 
        return True
    if ext=='.png': 
        return True
    if ext=='.jpg': 
        return True
    return False

def filter_for_images(fileList):
    image_list = []
    for i in range(len(fileList)):
        fileName = fileList[i]
        if isImageFile(fileName):
            image_list.append(fileName)
    return image_list

def find_all_images(dir_path = './'):
    ls_files = os.listdir(dir_path)
    print("There are %d files found in total."%len(ls_files))
    image_list = filter_for_images(ls_files)
    print("in which, %d image files are found."%len(image_list))
    for i in range(len(image_list)):
        image_list[i] =os.path.join(dir_path,image_list[i])
    return image_list

def getBaseNames(fileList):
    fileBaseNames=[]
    for i in range(len(fileList)):
        fileName = fileList[i]
        fileBaseNames.append(os.path.splitext(os.path.basename(fileName))[0])
    return fileBaseNames

def filter_CamVid_ImagePair(imgList,lblList):
    outImgList=[]
    outLblList=[]
    imgBaseNames = getBaseNames(imgList)
    lblBaseNames = getBaseNames(lblList)
    for i in range(len(imgBaseNames)):
        filename = imgBaseNames[i]
        supposedName = filename+"_L"
        if supposedName in lblBaseNames:
            j=lblBaseNames.index(supposedName) 
            outImgList.append(imgList[i])
            outLblList.append(lblList[j])
    return (outImgList,outLblList)



##### Label Txt file related
def pase_Label_File(filePath):
    f = open(filePath,mode='r')
    lines = f.readlines()
    label_items=[]
    print("Parsing txt file for labels...")
    for i in range(len(lines)):
        item = lines[i]
        item = item.split()
        r = np.uint8(item[0])
        g = np.uint8(item[1])
        b = np.uint8(item[2])
        name = (item[3])
        #print((r,g,b,name))
        label_items.append((r,g,b,name))
    print("There are %d classes in total"%len(label_items))
    f.close()
    return label_items

#####useful function
def readLabelTxt(filePath):
    basename_ext = os.path.splitext(filePath)
    ext = basename_ext[1].lower()
    label_items = None
    if ext!='.txt':
        raise ValueError('This is not txt file')
    else:
        label_items = pase_Label_File(filePath)
    return label_items  

def findLabelColorofName(label_items,name):
    for i in range(len(label_items)):
        item = label_items[i]
        if(item[3]==name): return (item[0],item[1],item[2])
    return None

def findLabelNameofColor(label_items,color):
    for i in range(len(label_items)):
        item = label_items[i]
        if((item[0],item[1],item[2])==color or [item[0],item[1],item[2]]==color): return item[3]
    return None

def getCamVidDataPairsList(imageDir,labelDir):
    print("Looking for images files...")
    imgList = find_all_images(imageDir)
    print("Looking for label files...")
    lblList = find_all_images(labelDir)
    print("Filtering for image correct label pairs...")
    ImgList,LblList = filter_CamVid_ImagePair(imgList,lblList)
    assert(len(ImgList)==len(LblList))
    print("There are %d image pairs in total"%len(ImgList))
    return ImgList,LblList

def transformLblImage(LBLImg,colorList):
    sp = np.shape(LBLImg)
    height = sp[0]
    width = sp[1]
    voidClrIdx=-1
    if((0,0,0) in colorList):
        voidClrIdx = colorList.index((0,0,0))
    elif([0,0,0] in colorList):
        voidClrIdx = colorList.index([0,0,0])
    else:
        raise ValueError("Cannot find Void color (0,0,0)")
    outPutLbl = np.zeros([height,width],dtype = np.int32)
    
    for i in range(height):
        for j in range(width):
            val = tuple(LBLImg[i][j])
            if (val in colorList):
                outPutLbl[i][j] = colorList.index(val)
            else:
                outPutLbl[i][j] = voidClrIdx
    return outPutLbl

def transformImgPairsToTrainableType(Img,Lbl,colorList):
    outImg = Img.astype(np.float32)
    outLbl = transformLblImage(Lbl,colorList)
    return outImg,outLbl

def getImgLblDataPair(ImgList,LblList,index):
    if(index>=len(ImgList)):
        raise ValueError("index out of range.")
    imgFilePath = ImgList[index]
    lblFilePath = LblList[index]
    IMG = np.array(Image.open(imgFilePath))
    LBL = np.array(Image.open(lblFilePath))
    return (IMG,LBL)

class TrainingParams:
    _batchsize = 0
    _epochs = 0
    _lr = 0
    _useRandomShuffle = True
    #getter
    def get_lr(self):
        return self._lr
    def get_epochs(self):
        return self._epochs
    def get_batchSize(self):
        return self._batchsize
    def get_UseRandomShuffle(self):
        return self._useRandomShuffle
    #setter
    def set_lr(self,lr):
        self._lr = lr
    def set_epochs(self,epochs):
        self._epochs = epochs
    def set_batchSize(self,batchSize):
        self._batchsize = batchSize
    def set_UseRandomShuffle(self,useRandomShuffle):
        self._useRandomShuffle = useRandomShuffle

class BatchDataReader:
    _batch_offset = 0
    _epochs_completed = 0
    _bRandomShuffle = True
    _ImgList=[]
    _LblList=[]
    _LblItems=None
    _totalFileNum=0
    _trainingParams=None
    _colorList = []
    _indexListForReader=[]
    def __init__(self,imageDir,labelDir,labelTxtFilePath,dataPairListGetter,trainingParams):
        print("Initializing Batch Dataset Reader...")
        self._ImgList,self._LblList = dataPairListGetter(imageDir,labelDir)
        self._totalFileNum = len(self._ImgList)
        self._trainingParams = trainingParams
        self._LblItems = readLabelTxt(labelTxtFilePath)
        self._colorList = self.getColorList(self._LblItems)
        self._batch_offset = 0
        self._indexListForReader = np.arange(self._totalFileNum)
        assert(len(self._ImgList)==len(self._LblList))
        assert(self._totalFileNum>0)
        assert(self._LblItems!=None)
        assert(isinstance(self._trainingParams,TrainingParams))
    
    def getSampleNum(self):
        return self._totalFileNum
        
    def getListPairs(self):
        return (self._ImgList,self._LblList)
    
    def reset_batch_offset(self, offset=0):
        self._batch_offset = offset
        
    def getColorList(self,LblItems):
        colorList=[]
        for i in range(len(LblItems)):
            colorList.append((LblItems[i][0],LblItems[i][1],LblItems[i][2]))
        return colorList
    
    def begin_epoch(self):
        param = self._trainingParams
        #if(self._epochs_completed<param.get_epochs()):
        self.reset_batch_offset();
        
        if(param.get_UseRandomShuffle()):
            self._indexListForReader = np.arange(self._totalFileNum)
            np.random.shuffle(self._indexListForReader)  
        else:
            self._indexListForReader = np.arange(self._totalFileNum)
        #return True
       # else:
         #   print("All epoches finished")
         #   return False
    
    def next_batch(self):
        batch_size = self._trainingParams.get_batchSize()
        totalSamplesNum = self._totalFileNum
        batch_img =[]
        batch_lbl = []
        for i in range(self._batch_offset,self._batch_offset+batch_size):
            if i>=totalSamplesNum:
                return None,None
                break
            print("sample %d"%i)
            img,lbl = getImgLblDataPair(self._ImgList,self._LblList,self._indexListForReader[i])
            img,lbl = transformImgPairsToTrainableType(img,lbl,self._colorList)
            batch_img.append(img)
            batch_lbl.append(lbl)
            
        self._batch_offset+=batch_size
        return np.array(batch_img),np.array(batch_lbl)
    
    def show_batch_image_pair(self,batch_img,batch_label,idxRange):
        num = len(idxRange)
        figsize = 4
        plt.figure(figsize=(2.3*figsize,num*figsize))
        for i in range(num):
            plt.subplot(num,2,i*2+1)
            self.show_image(batch_img[idxRange[i]])
            plt.subplot(num,2,i*2+2)
            self.show_labelImg(batch_label[idxRange[i]])
        plt.tight_layout()
    
    def show_image(self,img):
        plt.imshow(img.astype(np.uint8))
        
    def show_labelImg(self,lbl_img):
        sp = np.shape(lbl_img)
        height = sp[0]
        width = sp[1]
        showIMG = np.zeros([height,width,3],dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                index = lbl_img[i][j]
                clr = self._colorList[index]
                showIMG[i][j]=[clr[0],clr[1],clr[2]]
        plt.imshow(showIMG)



























































