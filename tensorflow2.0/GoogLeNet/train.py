from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import GoogLeNet
import tensorflow as tf
import json
import os

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
image_path = data_root + "/data_set/flower_data/"  # flower data set path
train_dir = image_path + "train"
validation_dir = image_path + "val"

# create direction for saving weights
if not os.path.exists("save_weights"):
    os.makedirs("save_weights")

im_height = 224
im_width = 224
batch_size = 32
epochs = 30


# 这里专门定义了一个对载入图像进行预处理的函数，作为实例化ImageDataGenerator类的参数
def pre_function(img):
    # img = im.open('test.jpg')
    # img = np.array(img).astype(np.float32)
    img = img / 255.
    img = (img - 0.5) * 2.0

    return img


# data generator with data augmentation
train_image_generator = ImageDataGenerator(preprocessing_function=pre_function,
                                           horizontal_flip=True)
# 除了单独定义一个预处理函数外，还可以直接将预处理方法写在参数中，如下：
# train_image_generator = ImageDataGenerator(rescale=((1. / 255) - 0.5) / 0.5 ,
#                                            horizontal_flip=True)

validation_image_generator = ImageDataGenerator(preprocessing_function=pre_function)

train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(im_height, im_width),
                                                           class_mode='categorical')
total_train = train_data_gen.n

# get class dict
class_indices = train_data_gen.class_indices

# transform value and key of dict
inverse_dict = dict((val, key) for key, val in class_indices.items())
# write dict into json file
json_str = json.dumps(inverse_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

val_data_gen = train_image_generator.flow_from_directory(directory=validation_dir,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         target_size=(im_height, im_width),
                                                         class_mode='categorical')
total_val = val_data_gen.n

model = GoogLeNet(im_height=im_height, im_width=im_width, class_num=5, aux_logits=True)
# model.build((batch_size, 224, 224, 3))  # when using subclass model
model.summary()
# GooLeNet一共大约有1000多万个参数，然后两个辅助分类器就占了400万个参数，所以最终模型的参数量是600万个

# using keras low level api for training
# 这里之所以没有使用keras提供的高级API(model.compile和model.fit)，是因为网络输出有三个，也就是在fit方法中要提供三组label构成的元组，但是ImageDataGenerator得到的label只有一组，所以这里不适合使用高级API训练网络
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        aux1, aux2, output = model(images, training=True)
        loss1 = loss_object(labels, aux1)
        loss2 = loss_object(labels, aux2)
        loss3 = loss_object(labels, output)
        loss = loss1*0.3 + loss2*0.3 + loss3
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, output)


@tf.function
def test_step(images, labels):
    _, _, output = model(images, training=False)
    t_loss = loss_object(labels, output)

    test_loss(t_loss)
    test_accuracy(labels, output)


best_test_loss = float('inf')
for epoch in range(1, epochs+1):
    train_loss.reset_states()        # clear history info
    train_accuracy.reset_states()    # clear history info
    test_loss.reset_states()         # clear history info
    test_accuracy.reset_states()     # clear history info

    for step in range(total_train // batch_size):
        images, labels = next(train_data_gen)
        train_step(images, labels)

    for step in range(total_val // batch_size):
        test_images, test_labels = next(val_data_gen)
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
    if test_loss.result() < best_test_loss:
        best_test_loss = test_loss.result()
        model.save_weights("./save_weights/myGoogLeNet.h5")
        # 在cpu训练的脚本中将模型的参数保存成.h5格式的文本
