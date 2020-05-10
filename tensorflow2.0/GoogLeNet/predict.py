from model import GoogLeNet
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt

im_height = 224
im_width = 224

# load image
img = Image.open("../tulip.jpg")
# resize image to 224x224
img = img.resize((im_width, im_height))
plt.imshow(img)

# scaling pixel value and normalize
img = ((np.array(img) / 255.) - 0.5) / 0.5

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))

# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

model = GoogLeNet(class_num=5, aux_logits=False)
# 在训练过程中，用了sequential和functional方式构建模型，所以模型是一个static graph，当做validation的时候，不能动态控制附加输出的分支，但是做inference的时候构建的模型不含附加输出的分支
model.summary()

# 模型载入参数的两种方式，对于h5格式，需要指定按照卷积层的名称载入参数；对于ckpt格式，则不需要指定，会自动按照名称给参数赋值
# model.load_weights("./save_weights/myGoogLenet.h5", by_name=True)  # h5 format
model.load_weights("./save_weights/myGoogLeNet.ckpt")  # ckpt format
result = model.predict(img)
# 预测结果应该是：
# result = np.squeeze(model.predict(img))
predict_class = np.argmax(result)
print(class_indict[str(predict_class)])
plt.show()
