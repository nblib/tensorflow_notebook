# coding:utf-8


from model.vgg.vgg16 import VGG16
from tensorflow.keras import datasets
import tensorflow as tf

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def _laod_and_preprocess():
    """
    加载和预处理数据,返回train:[50000,32,32,3], test:[10000,32,32,3]
    :return:
    """
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    return (train_images, train_labels), (test_images, test_labels)


if __name__ == '__main__':
    # load_data
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    (train_images, train_labels), (test_images, test_labels) = _laod_and_preprocess()
    model = VGG16([32, 32, 3], include_top=True, classes=len(class_names))
    model.summary()

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_images, train_labels,batch_size=32, epochs=20,
              validation_data=(test_images, test_labels))


