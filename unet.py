# 导入相应的库
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import cv2
from glob import glob

inline
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# 设置数据集路径
image_path = os.path.join("../input/chest-xray-masks-and-labels/Lung Segmentation/CXR_png/")
mask_path = os.path.join("../input/chest-xray-masks-and-labels/Lung Segmentation/masks/")

# 读取图片
images = os.listdir(image_path)
mask = os.listdir(mask_path)
mask = [fName.split(".png")[0] for fName in mask]
testing_files = set(os.listdir(image_path)) & set(os.listdir(mask_path))
training_files = [i for i in mask if "mask" in i]


# 处理训练集和测试集图片函数
def getData(X_shape, flag="test"):
    im_array = []
    mask_array = []

    if flag == "test":
        for i in tqdm(testing_files):
            im = cv2.resize(cv2.imread(os.path.join(image_path, i)), (X_shape, X_shape))[:, :, 0]
            mask = cv2.resize(cv2.imread(os.path.join(mask_path, i)), (X_shape, X_shape))[:, :, 0]

            im_array.append(im)
            mask_array.append(mask)

        return im_array, mask_array

    if flag == "train":
        for i in tqdm(training_files):
            im = cv2.resize(cv2.imread(os.path.join(image_path, i.split("_mask")[0] + ".png")), (X_shape, X_shape))[:,
                 :, 0]
            mask = cv2.resize(cv2.imread(os.path.join(mask_path, i + ".png")), (X_shape, X_shape))[:, :, 0]

            im_array.append(im)
            mask_array.append(mask)

        return im_array, mask_array


# 设置图片大小，加载训练集和测试集
dim = 512
X_train, y_train = getData(dim, flag="train")
X_test, y_test = getData(dim)

# 将训练集和测试集的图片进行预处理，然后进行数据合并
X_train = np.array(X_train).reshape(len(X_train), dim, dim, 1)
y_train = np.array(y_train).reshape(len(y_train), dim, dim, 1)
X_test = np.array(X_test).reshape(len(X_test), dim, dim, 1)
y_test = np.array(y_test).reshape(len(y_test), dim, dim, 1)
images = np.concatenate((X_train, X_test), axis=0)
mask = np.concatenate((y_train, y_test), axis=0)


# 展示数据集函数
def plotMask(X, y):
    sample = []

    for i in range(6):
        left = X[i]
        right = y[i]
        combined = np.hstack((left, right))
        sample.append(combined)

    for i in range(0, 6, 3):
        plt.figure(figsize=(25, 10))

        plt.subplot(2, 3, 1 + i)
        plt.imshow(sample[i])

        plt.subplot(2, 3, 2 + i)
        plt.imshow(sample[i + 1])

        plt.subplot(2, 3, 3 + i)
        plt.imshow(sample[i + 2])

        plt.show()


# 训练集和测试集分别展示六张图片
print("training set")
plotMask(X_train, y_train)
print("testing set")
plotMask(X_test, y_test)


# 定义损失函数
def dice_coef(y_true, y_pred):
    y_truef = K.flatten(y_true)
    y_predf = K.flatten(y_pred)
    And = K.sum(y_truef * y_predf)
    return (2 * And + 1) / (K.sum(y_truef) + K.sum(y_predf) + 1)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# 搭建U-Net模型
def unet(input_size=(256, 256, 1)):
    inputs = tf.keras.Input(input_size)

    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return tf.keras.Model(inputs=[inputs], outputs=[conv10])


# 创建模型保存文件夹
if not os.path.exists("save_weights"):
    os.makedirs("save_weights")

# 编译模型
model = unet(input_size=(512, 512, 1))
model.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-4), loss=dice_coef_loss,
              metrics=[dice_coef, 'binary_accuracy'])

# 打印模型参数
model.summary()

# 设置训练参数
checkpoint = ModelCheckpoint(filepath='./save_weights/myUnet.ckpt', monitor='val_loss', verbose=1,
                             save_best_only=True, mode='auto', save_weights_only=True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                   patience=3,
                                   verbose=1, mode='auto', epsilon=0.0001, cooldown=2, min_lr=1e-6)
early = EarlyStopping(monitor="val_loss",
                      mode="auto",
                      patience=20)
callbacks_list = [checkpoint, early, reduceLROnPlat]

# 将整合后的数据重新划分为训练集，验证集和测试集
train_vol, test_vol, train_seg, test_seg = train_test_split((images - 127.0) / 127.0,
                                                            (mask > 127).astype(np.float32),
                                                            test_size=0.1, random_state=2020)

train_vol, validation_vol, train_seg, validation_seg = train_test_split(train_vol, train_seg,
                                                                        test_size=0.1,
                                                                        random_state=2020)
# 开始训练
history = model.fit(x=train_vol,
                    y=train_seg,
                    batch_size=16,
                    epochs=50,
                    validation_data=(validation_vol, validation_seg),
                    callbacks=callbacks_list)
# 保存模型
model.save_weights('./save_weights/myUnet.ckpt', save_format='tf')

# 记录训练的损失值和准确率
history_dict = history.history
train_loss = history_dict["loss"]
train_accuracy = history_dict["binary_accuracy"]
val_loss = history_dict["val_loss"]
val_accuracy = history_dict["val_binary_accuracy"]

# 绘制损失值曲线
plt.figure()
plt.plot(range(50), train_loss, label='train_loss')
plt.plot(range(50), val_loss, label='val_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')

# 绘制准确率曲线
plt.figure()
plt.plot(range(50), train_accuracy, label='train_accuracy')
plt.plot(range(50), val_accuracy, label='val_accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

# 抽取测试集3张图片进行预测，并进行比较
pred_candidates = np.random.randint(1, validation_vol.shape[0], 10)
preds = model.predict(validation_vol)

plt.figure(figsize=(20, 10))

for i in range(0, 9, 3):
    plt.subplot(3, 3, i + 1)

    plt.imshow(np.squeeze(validation_vol[pred_candidates[i]]))
    plt.title("Base Image")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 3, i + 2)
    plt.imshow(np.squeeze(validation_seg[pred_candidates[i]]))
    plt.title("Mask")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 3, i + 3)
    plt.imshow(np.squeeze(preds[pred_candidates[i]]))
    plt.title("Pridiction")
    plt.xticks([])
    plt.yticks([])
