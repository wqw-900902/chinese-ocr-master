# coding:utf-8
from recognition_model import ctpn_model
from glob import glob
import numpy as np
from PIL import Image
import time
paths = glob('./test/*.*')


if __name__ == '__main__':

    img = Image.open('test/images/4788.png')
    print("原始 img shape:", np.array(img.convert('RGB')).shape)
    t = time.time()
    # 使用keras版本的模型直接预测出result结果
    result, img, angle = ctpn_model.model(img, model='pytorch')

    print("It takes time:{}s".format(time.time()-t))
    print("---------------------------------------")
    for key in result:
        print(result[key][1])
