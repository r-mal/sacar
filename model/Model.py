import tensorflow as tf

import numpy as np

import attribute
import concept
import relation

PREDICT = tf.estimator.ModeKeys.PREDICT
EVAL = tf.estimator.ModeKeys.EVAL
TRAIN = tf.estimator.ModeKeys.TRAIN


def model_fn(features, labels, mode, params):
  """
  Description...

  concept mentions are given during training, but not during testing
  """
  predictions = {}
  loss = None
  train_op = None
  eval_metric_ops = {}

  # concept boundary forward pass
  # save base encoding for attr and relation prediction
  boundary_logits, features['base_encoding'] = concept.prediction(features, params, mode)
  boundary_predictions = tf.argmax(boundary_logits, axis=-1)
  # use predicted concepts for attribute/relation detection

  # if mode == PREDICT or params.hard_eval:
  #   concept.boundary_predictions_to_labels(boundary_predictions, features, params)

  # attribute forward pass
  attr_logits, attr_predictions = attribute.prediction(features, params, mode)

  # relation forward pass
  relation_logits, relation_predictions = relation.prediction(features, params, mode)

  # calculate loss
  if mode in (TRAIN, EVAL):
    # boundary
    boundary_loss = concept.loss(features, labels, boundary_logits, params)

    # attr
    attr_loss = attribute.loss(features, labels, attr_logits, params)

    # relation
    relation_loss = relation.loss(features, labels, relation_logits, params)

    loss = params.boundary_loss_param * boundary_loss \
           + params.attr_loss_param * attr_loss \
           + params.rel_loss_param * relation_loss

  # get train _op
  if mode == TRAIN:
    step = tf.train.get_or_create_global_step()
    lr = tf.train.linear_cosine_decay(params.learning_rate, step, 1000)
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(loss, step)

  # perform predictions
  if mode == PREDICT:
    if params.mode == 'viz':
      predictions['record_id'] = features['record_id']
      # ATTN
      predictions['sentences'] = tf.expand_dims(tf.gather(tf.squeeze(features['words']), tf.squeeze(features['sentence_idxs'])), axis=0)
      print(predictions['sentences'])
      predictions['sentence_lens'] = features['sentence_lens']
      predictions['attn_weights'] = attribute.get_attention_weights()


      # # ACT
      # predictions['words ']= features['words']
      # predictions['doclen'] = features['doclen']
      # predictions['ponder_times'] = features['ponder_times']
      # predictions['boundary_labels'] = features['boundary_labels']
    else:
      if 'boundary' in params.modules:
        predictions['boundary'] = boundary_predictions
      if 'attr' in params.modules:
        for key, value in attr_predictions.iteritems():
          predictions['attr-%s' % key] = value
      #   if params.mode == 'viz':
      #     predictions['sent_attn_weights'] = attribute.get_attention_weights()
      if 'relation' in params.modules:
        predictions['relation'] = relation_predictions

  # perform evaluations
  if mode == EVAL:
    # boundary evals
    concept.evaluation(features, labels, boundary_predictions, params, eval_metric_ops)

    # attr evals
    attribute.evaluation(features, labels, attr_predictions, params, eval_metric_ops)

    # relation evals
    relation.evaluation(features, labels, relation_predictions, params, eval_metric_ops)
  
  # if mode == TRAIN:
  #   num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
  #   print("Num params: %d" % num_params)
  #   exit()

  return tf.estimator.EstimatorSpec(
    mode=mode,
    predictions=predictions,
    loss=loss,
    train_op=train_op,
    eval_metric_ops=eval_metric_ops
  )
