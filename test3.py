import os
import random
# 这个就是将annotations中的xml文件读取生成txt文件(所有图片的id，没有.jpg)，然后划分为三个文件夹train，val，test
# 这个就是划分的比例,本应该是这样，test占比1-0.9，然后train和val占比0.9，同时train占比train+val的0.9
# trainval_percent = 0.9
# train_percent = 0.9

# 我不需要test所以我设置为1.0
trainval_percent = 1.0
train_percent = 0.9
xmlfilepath = 'E:/data/Annotations'
txtsavepath = 'E:/data/ImageSets'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open('E:/data/ImageSets/trainval.txt', 'w+')
ftest = open('E:/data/ImageSets/test.txt', 'w+')
ftrain = open('E:/data/ImageSets/train.txt', 'w+')
fval = open('E:/data/ImageSets/val.txt', 'w+')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
