# 记得先修改为自己的路径
# okdata里面最好是训练集中的照片，100张左右
# calib_640_f32是生成的校准数据集
from pathlib import Path
import cv2
import numpy as np
import random


path = Path('./okdata')
save = Path('./calib_640_f32')
h = 640
w = 640

save.mkdir(parents=True, exist_ok=True)
cnt = 0
all_dirs = [i for i in path.iterdir()]
random.shuffle(all_dirs)
for i in all_dirs:
    if cnt >= 50:
        break
    img = cv2.imread(str(i))
    img = cv2.resize(img, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.astype(np.float32).tofile(str(save / (i.stem + '.rgbchw')))
    cnt += 1
    
