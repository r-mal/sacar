import tensorflow as tf
import transformers

PREDICT = tf.estimator.ModeKeys.PREDICT
EVAL = tf.estimator.ModeKeys.EVAL
TRAIN = tf.estimator.ModeKeys.TRAIN


def model_fn(features, labels, mode, params):
  predictions = {}
  loss = None
  train_op = None
  eval_metric_ops = {}

  words = features['tokens']

  with tf.variable_scope('base_transformer'):
    a_emb = tf.get_variable('a_emb', shape=[1, 1, params.embedding_size])
    b_emb = tf.get_variable('b_emb', shape=[1, 1, params.embedding_size])
    input_mask = tf.expand_dims(features['a_mask'], axis=-1) * a_emb \
                 + tf.expand_dims(features['b_mask'], axis=-1) * b_emb

    tparams = transformers.base_transformer_params(params)
    base_transformer = transformers.create(params, tparams, mode)

    # attn_bias = transformers.model_utils.get_padding_bias(words)
    base_word_encoding = base_transformer.encode(words, input_mask=input_mask)
    if params.transformer_type == 'adaptive':
      base_word_encoding, (features['ponder_times'], features['remainders']) = base_word_encoding

  # calculate loss
  if mode in (TRAIN, EVAL):
    # sentence selection
    cls = base_word_encoding[:, 0]
    sentence_selection_logits = tf.squeeze(tf.layers.dense(cls, 2, activation=tf.nn.sigmoid))
    sentence_selection_logprobs = tf.nn.log_softmax(sentence_selection_logits)
    sentence_selection_labels = tf.one_hot(tf.reshape(labels['label'], [-1]), 2, dtype=tf.float32)
    sentence_selection_loss = tf.reduce_mean(
      -tf.reduce_sum(sentence_selection_labels * sentence_selection_logprobs, axis=-1))
    tf.summary.scalar('sentence_selection_loss', sentence_selection_loss)
    # eval_metric_ops['sentence_selection_loss'] = sentence_selection_loss

    # token lm loss
    token_logits = tf.layers.dense(base_word_encoding, params.vocab_size, activation=tf.nn.relu)
    token_logprobs = tf.nn.log_softmax(token_logits)
    token_label_1hot = tf.one_hot(labels['lm_labels'], params.vocab_size, dtype=tf.float32)
    token_logprobs = -tf.reduce_sum(token_logprobs * token_label_1hot * tf.expand_dims(features['lm_mask'], axis=-1),
                                    axis=-1)
    masked_lm_loss = tf.reduce_mean(token_logprobs)
    tf.summary.scalar('masked_lm_loss', masked_lm_loss)
    # eval_metric_ops['masked_lm_loss'] = masked_lm_loss

    loss = sentence_selection_loss + masked_lm_loss
    tf.summary.scalar('loss', loss)
    # eval_metric_ops['loss'] = loss

  if mode == TRAIN:
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())


  return tf.estimator.EstimatorSpec(
    mode=mode,
    predictions=predictions,
    loss=loss,
    train_op=train_op,
    eval_metric_ops=eval_metric_ops
  )
