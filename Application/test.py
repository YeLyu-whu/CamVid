#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:21:44 2017

@author: yelyu
"""

import imp
import ImageReader
imp.reload(ImageReader)
from ImageReader import BatchDataReader

trainingParams = ImageReader.TrainingParams()
trainingParams.set_batchSize(10)
trainingParams.set_epochs(2)
trainingParams.set_lr(1.0)
trainingParams.set_UseRandomShuffle(True)
dataPairListGetter = ImageReader.getCamVidDataPairsList
reader = BatchDataReader("../701_StillsRaw_full","../LabeledApproved_full",
                         '../label_colors.txt',dataPairListGetter,
                         trainingParams)
reader.begin_epoch()
(batch_img,batch_lbl) = reader.next_batch()