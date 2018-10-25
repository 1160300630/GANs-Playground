import json
import tensorflow as tf
import numpy as np

def read_comic_test_label(file_name):
    with open('./dataset/txt2label.json', 'r') as f:
        text2label = json.load(f)
        
    labels = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            label = np.zeros(22, dtype=np.float32)
            line = line.split(',')[1].split(' ')
            hair = line[0] + ' ' + line[1]
            eyes = line[2] + ' ' + line[3]
            label[text2label[hair]] += 1
            label[text2label[eyes[:-1]]] += 1
            labels.append(label)
    return labels


def read_and_decode(file_name):
    filename_queue = tf.train.string_input_producer([file_name])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string)
    })
    
    label = tf.decode_raw(features['label'], tf.float32)
    image = tf.decode_raw(features['image'], tf.float32)
    
    image.set_shape([64*64*3])
    image = tf.reshape(image, [64, 64, 3])
    label.set_shape([22])
    label = tf.reshape(label, [22])
    return image, label