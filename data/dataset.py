import tensorflow as tf
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def load(config, record_file):
  # https://www.tensorflow.org/guide/datasets#preprocessing_data_with_datasetmap
  def record2tensors(example_proto):
    features_spec = {
      'record_id': tf.FixedLenFeature((), tf.string, default_value=''),
      'words': _feature([config.max_doc_len]),
      'doclen': _feature([1]),
      'sentence_idxs': _feature([config.sents_per_doc, config.toks_per_sent]),
      'sentence_lens': _feature([config.sents_per_doc]),
      'activity_idxs': _feature([config.sents_per_doc, config.concepts_per_sent]),
      'activity_lens': _feature([config.sents_per_doc]),
      'concept_idxs': _feature([config.sents_per_doc, config.concepts_per_sent]),
      'concept_lens': _feature([config.sents_per_doc]),
      'concept_mention_idxs': _feature([config.concept_mentions_per_doc]),
      'mentions_per_concept': _feature([config.concepts_per_doc, config.concepts_per_doc, config.mention_pairs, 2]),
      'mention_pairs_per_conc_pair': _feature([config.concepts_per_doc, config.concepts_per_doc]),
      'relation_mask': _feature([config.concepts_per_doc, config.concepts_per_doc]),
      'boundary_labels': _feature([config.max_doc_len]),
      'relation_labels': _feature([config.concepts_per_doc, config.concepts_per_doc])
    }
    for attr_name in config.attr_info.attr_label_maps.keys():
      features_spec[attr_name] = _feature([config.sents_per_doc, config.concepts_per_sent])

    parsed_features = tf.parse_single_example(example_proto, features_spec)
    label_names = ['boundary_labels', 'relation_labels'] + config.attr_info.attr_label_maps.keys()
    features = {fname: tensor for fname, tensor in parsed_features.iteritems() if fname not in label_names}
    labels = {fname: tensor for fname, tensor in parsed_features.iteritems() if fname in label_names}
    features['boundary_labels'] = labels['boundary_labels']

    return features, labels

  return tf.data.TFRecordDataset([record_file], compression_type="GZIP") \
    .map(record2tensors, num_parallel_calls=2) \
    .shuffle(100) \
    .batch(config.batch_size, drop_remainder=True)


def _feature(shape):
  return tf.FixedLenFeature(shape, tf.int64, default_value=np.zeros(shape, dtype=np.int64))


def preprocess(config):
  # concept
  # words  [batch_size, doc_len]
  # doclens [batch_size]

  # attr
  # sentence_idxs [batch_size, num_sentences, words_per_sentence]
  # sentence_lens [batch_size, num_sentences]
  # activity_idxs [batch_size, num_sentences, concepts_per_sentence]
  # activity_lens [batch_size, num_sentences]
  # concept_idxs [batch_size, num_sentences, concepts_per_sentence]
  # concept_lens [batch_size, num_sentences]

  # rel
  # concept_mention_idxs [batch_size, num_concept_mentions]
  # mentions_per_concept  [batch_size, global_concepts, global_concepts, mention_pairs, 2]
  #   last dim is [source mention, dest mention] index list
  # relation_mask [batch_size, global_concepts, global_concepts]

  # labels
  # boundary [batch_size, doc_len]
  # <attr_name> [batch_size, num_sentences, concepts_per_sentence]
  # relation [batch_size, global_concepts, global_concepts]

  from tensorflow import python_io as pio
  import os
  from random import random

  word2id = config.word2id
  activity_attributes = config.attr_info.activity_attributes
  common_attributes = config.attr_info.common_attributes

  train_file = os.path.join(config.record_dir, config.train_filename)
  val_file = os.path.join(config.record_dir, config.val_filename)

  num_train, num_val = 0, 0
  with pio.TFRecordWriter(train_file, options=pio.TFRecordOptions(pio.TFRecordCompressionType.GZIP)) as twriter, \
    pio.TFRecordWriter(val_file, options=pio.TFRecordOptions(pio.TFRecordCompressionType.GZIP)) as vwriter:
    jdata = json.load(open(config.json_data))
    for json_doc in tqdm(jdata, total=len(jdata)):
      words = []
      slens = []
      word_idxs = np.zeros([config.sents_per_doc, config.toks_per_sent], int)
      activity_lens = []
      activity_idxs = np.zeros([config.sents_per_doc, config.concepts_per_sent], int)
      concept_lens = []
      concept_idxs = np.zeros([config.sents_per_doc, config.concepts_per_sent], int)
      boundary_labels = []
      attribute_labels = defaultdict(lambda: np.zeros([config.sents_per_doc, config.concepts_per_sent], int))
      sent_offset = 0

      for i, sentence in enumerate(json_doc['sentences'][:config.sents_per_doc]):
        swords = [word2id[word] for word in sentence['words'][:config.toks_per_sent]]
        slens.append(len(swords))
        words += swords
        boundary_labels += [config.boundary2id[b] for b in sentence['boundary_labels'][:config.toks_per_sent]]

        word_idxs[i, :slens[i]] = range(sent_offset, sent_offset + slens[i])
        sent_offset += slens[i]

        concepts = sentence['concepts'][:config.concepts_per_sent]
        alen = 0
        concept_lens.append(len(concepts))
        for j, (cidx, label_dict) in enumerate(concepts):
          if 'morphology' in label_dict:
            activity_idxs[i, alen] = cidx
            alen += 1
            for aname in activity_attributes:
              attribute_labels[aname][i, alen] = label_dict[aname]
          concept_idxs[i, j] = cidx
          for aname in common_attributes:
            attribute_labels[aname][i, j] = label_dict[aname]
        activity_lens.append(alen)
      words = words[:config.max_doc_len]
      doclen = len(words)

      # concept_mentions[i] = doc-level id of word corresponding to ith mention
      concept_mentions = np.zeros(config.concept_mentions_per_doc, int)
      # conc2mentions[cid] = [mid] list of mention ids (indexes into concept_mentions)
      conc2mentions = defaultdict(list)
      for i, (concept_id, mention_idxs) in enumerate(json_doc['relations']['mentions'].items()[:config.concept_mentions_per_doc]):
        if int(concept_id) < config.concepts_per_doc:
          for [sent_id, word_id] in mention_idxs:
            if sent_id < config.sents_per_doc and word_id < config.toks_per_sent:
              concept_mentions[i] = word_idxs[sent_id, word_id]
              conc2mentions[int(concept_id)].append(i)
      relation_mask = np.zeros([config.concepts_per_doc, config.concepts_per_doc], int)
      mentions_per_concept = np.zeros([config.concepts_per_doc, config.concepts_per_doc, config.mention_pairs, 2], int)
      mention_pairs_per_conc_pair = np.zeros([config.concepts_per_doc, config.concepts_per_doc], int)
      print('Found %d concepts with %d mentions' % (len(conc2mentions), sum([len(m) for m in conc2mentions.values()])))

      # populate relation mask, mention_pairs_per_conc_pair, and mentions_per_concept
      for head_cid, head_mentions in dict(conc2mentions).iteritems():
        for tail_cid, tail_mentions in dict(conc2mentions).iteritems():
          if head_cid != tail_cid:
            relation_mask[head_cid, tail_cid] = 1
            mention_pairs_per_conc_pair[head_cid, tail_cid] = \
              min(config.mention_pairs, len(conc2mentions[head_cid]) * len(conc2mentions[tail_cid]))
            for h, head_mention in enumerate(head_mentions):
              for t, tail_mention in enumerate(tail_mentions):
                if h + t >= config.mention_pairs:
                  break
                mentions_per_concept[head_cid, tail_cid, h + t] = [head_mention, tail_mention]
      print('Mask members: %d' % int(np.sum(relation_mask)))

      relation_labels = np.zeros([config.concepts_per_doc, config.concepts_per_doc], int)
      # populate relation_labels
      for label, head_cid, tail_cid in json_doc['relations']['labels']:
        relation_labels[head_cid, tail_cid] = label + 1
        # relation_mask[head_cid, tail_cid] = 1
        # mention_pairs_per_conc_pair[head_cid, tail_cid] = \
        #   min(config.mention_pairs, len(conc2mentions[head_cid]) * len(conc2mentions[tail_cid]))
        # for h, head_mention in enumerate(conc2mentions[head_cid]):
        #   for t, tail_mention in enumerate(conc2mentions[tail_cid]):
        #     if h + t >= config.mention_pairs:
        #       break
        #     mentions_per_concept[head_cid, tail_cid, h + t] = [head_mention, tail_mention]

      if len(conc2mentions) > 0:
        features = {
          'record_id': _bytes_feature(json_doc['id']),
          'words': _int64_feature(words + [0] * (config.max_doc_len - len(words))),
          'doclen': _int64_feature([doclen]),
          'sentence_idxs': _int64_feature(word_idxs.flatten().tolist()),
          'sentence_lens': _int64_feature(slens + [0] * (config.sents_per_doc - len(slens))),
          'activity_idxs': _int64_feature(activity_idxs.flatten().tolist()),
          'activity_lens': _int64_feature(activity_lens + [0] * (config.sents_per_doc - len(activity_lens))),
          'concept_idxs': _int64_feature(concept_idxs.flatten().tolist()),
          'concept_lens': _int64_feature(concept_lens + [0] * (config.sents_per_doc - len(concept_lens))),
          'concept_mention_idxs': _int64_feature(concept_mentions.tolist()),
          'mentions_per_concept': _int64_feature(mentions_per_concept.flatten().tolist()),
          'relation_mask': _int64_feature(relation_mask.flatten().tolist()),
          'mention_pairs_per_conc_pair': _int64_feature(mention_pairs_per_conc_pair.flatten().tolist()),
          'boundary_labels': _int64_feature(boundary_labels[:config.max_doc_len] +
                                            [config.boundary2id['O']] * (config.max_doc_len - len(boundary_labels))),
          'relation_labels': _int64_feature(relation_labels.flatten().tolist())
        }
        for attr_label, labels_matrix in attribute_labels.iteritems():
          features[attr_label] = _int64_feature(labels_matrix.flatten().tolist())

        example = tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
        if random() < config.validation_proportion and num_val < 36:
          vwriter.write(example)
          num_val += 1
        else:
          twriter.write(example)
          num_train += 1
  print("Saved %d training examples" % num_train)
  print("Saved %d validation examples" % num_val)


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))


class MapMapWrapper(object):
  def __init__(self, old_word2id, word2id):
    self.id2word = {v: k for k, v in old_word2id.items()}
    self.word2id = word2id

  def __getitem__(self, item):
    return self.word2id[self.id2word[item]]
