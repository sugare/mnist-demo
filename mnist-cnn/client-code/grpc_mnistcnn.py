import grpc_client
import tensorflow as tf
import random
import numpy as np
from tensorflow.python.framework import tensor_util
from PIL import Image
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets as input_data
tf.app.flags.DEFINE_string("host", "0.0.0.0", "TensorFlow Serving server ip")
tf.app.flags.DEFINE_integer("port", 8500, "TensorFlow Serving server port")
tf.app.flags.DEFINE_string("model_name", "default", "The model name")
tf.app.flags.DEFINE_string("signature_name", 'predict_images',
                           "The model signature name")
tf.app.flags.DEFINE_float("request_timeout", 5.0, "Timeout of gRPC request")
tf.app.flags.DEFINE_integer("num_tests", 100,
                            "number of predictions you want to make")
tf.app.flags.DEFINE_float("dropout", 0.5, "dropout of cnn")
FLAGS = tf.app.flags.FLAGS
mnist = input_data("/tmp/mnist-data", one_hot=True)
input_images = mnist.test.images
if FLAGS.num_tests % 10 != 0:
    FLAGS.num_tests = (FLAGS.num_tests / 10) * 10
client = grpc_client.Grpc_client(FLAGS.host, FLAGS.port)


def predict(l):
    curMax = 0
    index = 0
    for i in range(len(l)):
        if l[i] > curMax:
            index = i
            curMax = l[i]
    return index


def showImage(img):
    res = []
    for i in range(28):
        res.append([])
        for j in range(280):
            k = int(255 * img[i][j])
            res[i].append([k, k, k, 255])
    rawImg = np.array(res, dtype=np.uint8)

    return rawImg


predictionLabels = []
predictionImg = None
for _ in range(FLAGS.num_tests / 10):
    for _ in range(10):
        i = random.randint(0, 9999)
        request = client.set_request(
            name=FLAGS.model_name,
            signature_name=FLAGS.signature_name,
            inputs={
                'images':
                tf.contrib.util.make_tensor_proto(
                    input_images[i], shape=[1, input_images[0].size]),
                'keep_prob':
                tf.contrib.util.make_tensor_proto(FLAGS.dropout)
            })
        result = client.make_prediction(request, FLAGS.request_timeout)
        IMG = input_images[i].reshape(28, 28)
        try:
            IMGs = np.concatenate((IMGs, IMG), axis=1)
        except:
            IMGs = IMG

        predictionLabels.append(
            predict(tensor_util.MakeNdarray(result.outputs['scores'])[0]))
    try:
        predictionImg = np.concatenate(
            (predictionImg, showImage(IMGs)), axis=0)
    except:
        predictionImg = showImage(IMGs)
    IMGs = None

img = Image.fromarray(predictionImg)
img.show()
print(predictionLabels)
