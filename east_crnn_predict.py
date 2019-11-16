import time
from recognition_model import east_model as model


# AdvancedEAST模型进行OCR识别
if __name__ == '__main__':

    path = 'test/images/013.jpg'
    t = time.time()
    result = model.advancedEAST(path, "crnn")
    print("It takes time:{}s".format(time.time()-t))
    print("-----------------识别文本内容-------------------")
    for txt in result:
        print(txt)
