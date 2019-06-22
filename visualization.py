import tensorflow as tf
import os
import io

from model.Model import model_fn
from data.dataset import load


def save_act_data(config):
  config.batch_size = 1

  estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=config.model_dir,
    params=config
  )

  predictions = estimator.predict(input_fn=lambda: load(config, os.path.join(config.record_dir, config.val_filename)).make_one_shot_iterator().get_next())
  id2word = {v: k for k, v in config.word2id.iteritems()}
  id2boundary = {v: k for k, v in config.boundary2id.iteritems()}
  data = []
  ids = []
  maxdoclen = -1
  for i, prediction in enumerate(predictions):
    print('Example %d: %s' % (i, prediction['record_id']))
    ids.append(prediction['record_id'])
    doclen = prediction['doclen'][0]
    maxdoclen = doclen if doclen > maxdoclen else maxdoclen
    words = [id2word[w] if id2word[w] != ',' else '<COMMA>' for w in prediction['words'][:doclen]]
    labels = [id2boundary[w] for w in prediction['boundary_labels'][:doclen]]
    ponder_times = prediction['ponder_times'][:doclen]
    print('(Label, Word, PonderTime): %s' % '\n'.join([str(t) for t in zip(labels, words, ponder_times)]))
    data.append((labels, words, ponder_times))

  with io.open(os.path.join(config.model_dir, config.run_name, 'visualization2.csv'), 'w+', encoding='utf-8') as f:
    f.write(u','.join(ids))
    f.write(u'\n')
    for i in range(maxdoclen):
      for (labels, words, ponder_times) in data:
        print('i: %d, lbl: %d, w: %d, p: %d' % (i, len(labels), len(words), len(ponder_times)))
        if len(labels) > i and len(words) > i and len(ponder_times) > i:
          f.write(u'%s,%s,%d,,,' % (labels[i], words[i], ponder_times[i]))
      f.write(u'\n')


def save_attn_data(config):
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt
  import matplotlib as mpl

  mpl.rcParams.update({'figure.autolayout':True})
  # plt.tight_layout()


  config.batch_size = 1

  estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=config.model_dir,
    params=config
  )

  predictions = estimator.predict(input_fn=lambda: load(config, os.path.join(config.record_dir, config.val_filename)).make_one_shot_iterator().get_next())
  id2word = {v: k for k, v in config.word2id.iteritems()}
  outdir = os.path.join(config.model_dir, config.run_name, 'attention_plots')
  if not os.path.exists(outdir):
    os.mkdir(outdir)
  for i, prediction in enumerate(predictions):
    print('Example %d: %s' % (i, prediction['record_id']))
    if not os.path.exists(os.path.join(outdir, prediction['record_id'])):
      os.mkdir(os.path.join(outdir, prediction['record_id']))

    lens = prediction['sentence_lens']
    # print(lens.shape)
    sentences = prediction['sentences']
    # print(sentences.shape)
    attn_weights = prediction['attn_weights']
    # print(attn_weights.shape)

    for s, (slen, sentence, weights) in enumerate(zip(lens, sentences, attn_weights)):
      # weights [4, slen, slen]
      if slen > 0:
        words = [id2word[w] for w in sentence[:slen]]
        columns0 = {}
        columns1 = {}
        columns2 = {}
        columns3 = {}
        for w, word in enumerate(words):
          weight_vector = weights[:, w, :slen]  # [4, slen]
          columns0[word] = weight_vector[0]
          columns1[word] = weight_vector[1]
          columns2[word] = weight_vector[2]
          columns3[word] = weight_vector[3]
          # columns[word] = weight_vector[0, :slen]
        table = pd.DataFrame(columns0, index=words, columns=words)
        plotfile = os.path.join(outdir, prediction['record_id'], 'sent%d-0.png' % s)
        plot = sns.heatmap(table, vmin=0., xticklabels=True, yticklabels=True)
        fig = plot.get_figure()
        fig.savefig(plotfile)
        plt.clf()

        table = pd.DataFrame(columns1, index=words, columns=words)
        plotfile = os.path.join(outdir, prediction['record_id'], 'sent%d-1.png' % s)
        plot = sns.heatmap(table, vmin=0., xticklabels=True, yticklabels=True)
        fig = plot.get_figure()
        fig.savefig(plotfile)
        plt.clf()

        table = pd.DataFrame(columns2, index=words, columns=words)
        plotfile = os.path.join(outdir, prediction['record_id'], 'sent%d-2.png' % s)
        plot = sns.heatmap(table, vmin=0., xticklabels=True, yticklabels=True)
        fig = plot.get_figure()
        fig.savefig(plotfile)
        plt.clf()

        table = pd.DataFrame(columns3, index=words, columns=words)
        plotfile = os.path.join(outdir, prediction['record_id'], 'sent%d-3.png' % s)
        plot = sns.heatmap(table, vmin=0., xticklabels=True, yticklabels=True)
        fig = plot.get_figure()
        fig.savefig(plotfile)
        plt.clf()

        # _ = raw_input('Press enter to continue...')
