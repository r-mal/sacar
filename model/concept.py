import tensorflow as tf
import transformers


def prediction(features, params, mode):
  """
  TODO: enable pretrained model
  - add [cls], [sep] tokens
  - add a embedding
  :param features:
  :param params:
  :param mode:
  :return:
  """
  words = features['words']  # [batch_size, doc_len]

  with tf.variable_scope('base_transformer'):
    tparams = transformers.base_transformer_params(params)
    base_transformer = transformers.create(params, tparams, mode)

    # if pretrained, add leading [cls] and trailing [sep]
    if params.pretrained:
      # prepend [CLS] token
      words = tf.concat([tf.ones([words.shape[0], 1], tf.int64) * params.word2id['[CLS]'], words], axis=1)[:, :-1]
      # append [SEP] token to end of each doc
      word_len = seqlen(words)
      onehot = tf.one_hot(word_len, params.max_doc_len, dtype=tf.int64)  # [batch, doc_len]
      zerohot = tf.to_int64(tf.equal(onehot, 0))
      words = (zerohot * words) + (onehot * params.word2id['[SEP]'])
      # apply A sentence embedding
      attn_bias = tf.get_variable('a_emb', shape=[1, 1, params.embedding_size])
    else:
      attn_bias = None if params.transformer_type == 'adaptive' else transformers.model_utils.get_padding_bias(words)


    base_word_encoding = base_transformer.encode(words, attn_bias)
    if params.transformer_type == 'adaptive':
      base_word_encoding, (features['ponder_times'], features['remainders']) = base_word_encoding

  # print('ACT Graph nodes:')
  # for t in tf.get_default_graph().as_graph_def().node:
  #   if 'n_updates' in t.name:
  #     print(t.name)
  logits = tf.layers.dense(base_word_encoding, len(params.boundary2id), name='cb_logits')
  if mode == tf.estimator.ModeKeys.TRAIN:
    logits = tf.layers.dropout(logits, rate=params.drop_prob)

  return logits, base_word_encoding


def loss(features, labels, logits, params):
  if 'concept' in params.modules:
    boundary_eval_mask = tf.sequence_mask(tf.squeeze(features['doclen']), maxlen=params.max_doc_len, dtype=tf.float32)
    # weight non-O classes x10
    class_weights = (tf.to_float(tf.not_equal(labels['boundary_labels'], params.boundary2id['O'])) + 1.) * 10.
    boundary_eval_mask *= class_weights
    boundary_loss = tf.losses.sparse_softmax_cross_entropy(labels['boundary_labels'],
                                                           logits,
                                                           weights=boundary_eval_mask,
                                                           scope='boundary')
    tf.summary.scalar('boundary_loss', boundary_loss)
    if params.transformer_type == 'adaptive':
      act_loss = tf.reduce_mean(features['ponder_times'] + features['remainders'])
      tf.summary.scalar('boundary_act_loss', act_loss)
      boundary_loss += params.act_loss_param * act_loss
    return boundary_loss
  else:
    return 0.


def evaluation(features, labels, predictions, params, eval_metric_ops):
  boundary_eval_mask = tf.sequence_mask(tf.squeeze(features['doclen']), maxlen=params.max_doc_len, dtype=tf.float32)
  f1s = []
  for name, i in params.boundary2id.iteritems():
    labels_i = tf.equal(labels['boundary_labels'], i)
    predictions_i = tf.equal(predictions, i)
    r = tf.metrics.recall(labels_i, predictions_i, weights=boundary_eval_mask)
    p = tf.metrics.precision(labels_i, predictions_i, weights=boundary_eval_mask)
    f1 = f1_metric(p, r)
    if name != "O":
      f1s.append(f1[0])
    eval_metric_ops['%s-f1' % name] = f1

    if params.verbose_eval:
      eval_metric_ops['%s-acc' % name] = tf.metrics.accuracy(labels_i, predictions_i, weights=boundary_eval_mask)
      eval_metric_ops['%s-recall' % name] = r
      eval_metric_ops['%s-prec' % name] = p
  macro_avg_f1 = tf.reduce_mean(tf.stack(f1s))
  eval_metric_ops['boundary-f1'] = macro_avg_f1, macro_avg_f1  # tf.group(f1_update_ops)


def boundary_predictions_to_labels(boundary_predictions, features, params):
  # TODO: fill features['activity_labels'] and features['other_labels'] with predictions
  raise NotImplementedError()


def f1_metric(prec_metric, recall_metric):
  p, r = prec_metric[0], recall_metric[0]
  f1 = 2 * p * r / (p + r + 1e-5)
  return f1, tf.group(prec_metric[1], recall_metric[1])


def seqlen(idxs, dtype=tf.int32):
  return tf.reduce_sum(tf.cast(tf.greater(idxs, 0), dtype), axis=-1)
