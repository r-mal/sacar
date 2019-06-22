import argparse
import sys
import json
import os
from collections import defaultdict

import train
from data import dataset, pretrain
import visualization

run_switcher = {
    'train': train.main,
    'preprocess': dataset.preprocess,
    'eval': None,
    'viz': visualization.save_act_data,
    'pretrain-create': pretrain.preprocess,
    'pretrain': train.pretrain
  }


class AttrInfo(object):
  def __init__(self, attr_dir):
    print('Loading attribute info from %s...' % attr_dir)
    self.attr_label_maps = {}
    self.attr_weight_maps = {}
    self.common_attributes = {'polarity', 'modality'}
    self.activity_attributes = []
    for fname in os.listdir(attr_dir):
      if fname.endswith('.json'):
        attr = fname.replace('.json', '')
        self.attr_label_maps[attr] = json.load(open(os.path.join(attr_dir, fname)))
        if attr not in self.common_attributes:
          self.activity_attributes += [attr]
    weight_dir = os.path.join(attr_dir, 'attr_class_weights')
    for fname in os.listdir(weight_dir):
      if fname.endswith('.json'):
        attr = fname.replace('.json', '')
        self.attr_weight_maps[attr] = defaultdict(lambda: 0.)
        for k, v in json.load(open(os.path.join(weight_dir, fname))).items():
          self.attr_weight_maps[attr][k] = v
    self.activity_attributes = set(self.activity_attributes)


class DataInfo(object):
  def __init__(self, data_dir):
    self.record_dir = data_dir
    self.json_data = os.path.join(data_dir, 'data.json')
    self.word2id = json.load(open(os.path.join(data_dir, 'word2id.json')))
    self.boundary2id = json.load(open(os.path.join(data_dir, 'boundary2id.json')))
    self.rel2id = json.load(open(os.path.join(data_dir, 'rel2id.json')))


def load_or_default(field, default, loader=lambda x: x):
  # returns the loaded field if it is set or the default otherwise
  if field is None:
    return default
  else:
    return loader(field)


# noinspection PyTypeChecker
def main():
  parser = argparse.ArgumentParser(
    description='Train and Evaluate the Joint learning network for EEG knowledge extraction')

  # control options
  parser.add_argument('mode', choices=run_switcher.keys())
  parser.add_argument('--run_name', default=None, help='Name for the run')
  parser.add_argument('--verbose_eval', default=False, type=bool,
                      help='Print fine-grained evaluations (per class in addition to per-task)')
  parser.add_argument('--hard_eval', default=False, type=bool,
                      help='Evaluate attr/relation with predicted concept spans')
  parser.add_argument('--validation_proportion', default=0.2, type=float)
  parser.add_argument('--warm_start_model', default=None, help='model directory with saved model to use for warm start')
  parser.add_argument('--embedding_device', default='/device:GPU:0',
                      choices=['/device:GPU:0', '/device:GPU:1', '/cpu:0'])
  parser.add_argument('--pretrained', default=False, type=bool)

  # data
  parser.add_argument('--model_dir', default='/home/rmm120030/working/eeg/joint/model', help='model dir')
  parser.add_argument('--base_data_dir', default='/home/rmm120030/working/eeg/joint/data/io/',
                      help='Base data directory', type=DataInfo)
  parser.add_argument('--record_dir', default=None, help=('Location of directory the tfrecords should be in.' +
                                                          'If None, will use base_data_dir/data.json'))
  parser.add_argument('--train_filename', default='train.tfrecord', help='Train tfrecord filename')
  parser.add_argument('--val_filename', default='val.tfrecord', help='Validation tfrecord filename')
  parser.add_argument('--json_data', default=None, help=('Location of raw json data.' +
                                                         'If None, will use base_data_dir/data.json'))
  parser.add_argument('--word2id', default=None, help=('Json file with word to id mapping.' +
                                                       'If None, will use base_data_dir/word2id.json'))
  parser.add_argument('--boundary2id', default=None, help=('Json file with boundary_label to id mapping. ' +
                                                           'If None, will use base_data_dir/boundary2id.json'))
  parser.add_argument('--rel2id', default=None, help=('Json file with relation_label to id mapping. ' +
                                                      'If None, will use base_data_dir/rel2id.json'))
  parser.add_argument('--attr_info', default='/home/rmm120030/working/eeg/joint/data/attr', type=AttrInfo,
                      help='Directory with attribute to id mapping files.')
  parser.add_argument('--pretrain_data_dir', default='/home/rmm120030/working/eeg/joint/data/pretrain')

  # hyperparameters
  parser.add_argument('--learning_rate', default=1e-4, type=float)
  parser.add_argument('--batch_size', default=4, type=int)
  parser.add_argument('--drop_prob', default=0.1, type=float)
  parser.add_argument('--boundary_loss_param', default=1, type=float)
  parser.add_argument('--act_loss_param', default=0.1, type=float)
  parser.add_argument('--attr_loss_param', default=1, type=float)
  parser.add_argument('--rel_loss_param', default=1, type=float)
  parser.add_argument('--num_epochs', default=100, type=int)
  parser.add_argument('--steps_per_epoch', default=10, type=int)
  parser.add_argument('--class_weighting', default=False, type=bool)
  parser.add_argument('--multi_layer_attr', default=False, type=bool)
  parser.add_argument('--eval_delay', default=180, type=int)
  parser.add_argument('--eval_throttle', default=600, type=int)

  # parameters
  parser.add_argument('--modules', default=['concept', 'attr', 'relation'], nargs='+',
                      help='Modules to include: [concept, attr, relation]')
  parser.add_argument('--max_doc_len', default=519, type=int, help='Maximum number of tokens in a document')
  parser.add_argument('--vocab_size', default=21078, type=int, help='Number of unique tokens')  # 21074
  parser.add_argument('--embedding_size', default=256, type=int, help='Word embedding size')
  parser.add_argument('--base_transformer_num_heads', default=8, type=int,
                      help='Number of attention heads in the base transformer')
  parser.add_argument('--base_transformer_num_stacks', default=6, type=int,
                      help='Number of times the base transformer is stacked')
  parser.add_argument('--use_convolution', default=True, type=bool, help='Use convolutional layer in transformers?')
  parser.add_argument('--transformer_type', default='adaptive', choices=['transformer', 'universal', 'adaptive'],
                      help='transformer type for narrative/sentence encoders')
  # attr
  parser.add_argument('--sents_per_doc', default=25, type=int, help='Maximum number of sentences in a document')
  parser.add_argument('--toks_per_sent', default=65, type=int, help='Maximum number of tokens in a sentence')
  parser.add_argument('--concepts_per_sent', default=18, type=int, help='Maximum number of concepts in a sentence')
  parser.add_argument('--sentence_transformer_num_heads', default=4, type=int,
                      help='Number of attention heads in the sentence transformer')
  parser.add_argument('--sentence_transformer_num_stacks', default=2, type=int,
                      help='Number of times the sentence transformer is stacked')
  # rel
  parser.add_argument('--concept_mentions_per_doc', default=59, type=int,
                      help='Maximum number of concept mentions in a document')
  parser.add_argument('--concepts_per_doc', default=51, type=int,
                      help='Maximum number of unique concepts in a document')
  parser.add_argument('--mention_pairs', default=49, type=int, help='Maximum number of mention pairs in a document')
  parser.add_argument('--rel_hidden_size', default=100, type=int, help='Size of source/dest argument encodings')


  config = parser.parse_args(sys.argv[1:])

  config.record_dir = load_or_default(config.record_dir, config.base_data_dir.record_dir)
  config.json_data = load_or_default(config.json_data, config.base_data_dir.json_data)
  config.word2id = load_or_default(config.word2id, config.base_data_dir.word2id, loader=lambda x: json.load(open(x)))
  config.boundary2id = load_or_default(config.boundary2id, config.base_data_dir.boundary2id,
                                       loader=lambda x: json.load(open(x)))
  config.rel2id = load_or_default(config.rel2id, config.base_data_dir.rel2id, loader=lambda x: json.load(open(x)))

  run_switcher[config.mode](config)


if __name__ == "__main__":
  main()
