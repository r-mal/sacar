import tensorflow as tf
import json
import numpy as np
from tqdm import tqdm


def load(config, record_file):
  def record2tensors(example_proto):
    features_spec = {
      'label': _feature([1]),
      'tokens': _feature([config.toks_per_sent]),
      'a_mask': _float_feature([config.toks_per_sent]),
      'b_mask': _float_feature([config.toks_per_sent]),
      'lm_labels': _feature([config.toks_per_sent]),
      'lm_mask': _float_feature([config.toks_per_sent])
    }

    features = tf.parse_single_example(example_proto, features_spec)
    labels = {'label': features['label'],
              'lm_labels': features['lm_labels']}
    del features['label']
    del features['lm_labels']

    return features, labels

  return tf.data.TFRecordDataset([record_file], compression_type="GZIP") \
    .map(record2tensors, num_parallel_calls=4) \
    .shuffle(100) \
    .batch(config.batch_size)


def _feature(shape):
  return tf.FixedLenFeature(shape, tf.int64, default_value=np.zeros(shape, dtype=np.int64))


def _float_feature(shape):
  return tf.FixedLenFeature(shape, tf.float32, default_value=np.zeros(shape, dtype=np.float32))


def preprocess(config):
  from tensorflow import python_io as pio
  import os
  from random import random

  data_dir = config.pretrain_data_dir
  word2id = json.load(open(os.path.join(data_dir, 'word2id.json')))
  slen = config.toks_per_sent

  numval, numtrain = 0, 0

  with pio.TFRecordWriter(os.path.join(data_dir, 'train.tfrecord'),
                          options=pio.TFRecordOptions(pio.TFRecordCompressionType.GZIP)) as twriter, \
    pio.TFRecordWriter(os.path.join(data_dir, 'val.tfrecord'),
                       options=pio.TFRecordOptions(pio.TFRecordCompressionType.GZIP)) as vwriter:
    jdata = json.load(open(os.path.join(data_dir, 'data.json')))
    for example in tqdm(jdata, total=len(jdata)):
      tokens = [word2id['[CLS]']] + example['s1']['tokens'] + [word2id['[SEP]']] \
          + example['s2']['tokens'] + [word2id['[SEP]']]

      s1_len = len(example['s1']['tokens'])
      s2_len = len(example['s2']['tokens'])
      s1_labels, s1_mask = _labels_and_mask(example['s1']['labels'], s1_len)
      s2_labels, s2_mask = _labels_and_mask(example['s2']['labels'], s2_len)
      a_mask = [0.] + [1.]*s1_len + [0.]*(2 + s2_len)
      b_mask = [0.]*(2 + s1_len) + [1.]*s2_len + [0.]
      lm_labels = [0] + s1_labels + [0] + s2_labels + [0]
      lm_mask = [0.] + s1_mask + [0.] + s2_mask + [0.]

      features = {
        'label': _int64_feature([example['label']]),
        'tokens': _padded_int64_feature(tokens, slen),
        'a_mask': _padded_float_feature(a_mask, slen),
        'b_mask': _padded_float_feature(b_mask, slen),
        'lm_labels': _padded_int64_feature(lm_labels, slen),
        'lm_mask': _padded_float_feature(lm_mask, slen)
      }

      if random() > 0.05:
        numtrain += 1
        twriter.write(tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString())
      else:
        numval += 1
        vwriter.write(tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString())
  print('Done! Created %d training examples and %d validation examples' % (numtrain, numval))


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _padded_float_feature(value, list_len):
  pad = [0.] * (list_len - len(value[:list_len]))
  """Wrapper for inserting int64 features into Example proto."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value + pad))


def _padded_int64_feature(value, list_len):
  pad = [0] * (list_len - len(value[:list_len]))
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value + pad))


def _labels_and_mask(lmdict, list_len):
  labels = [0]*list_len
  mask = [0.]*list_len
  for idx, lbl in lmdict.iteritems():
    idx = int(idx)
    labels[idx] = lbl
    mask[idx] = 1.
  return labels, mask
