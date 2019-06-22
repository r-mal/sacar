import tensorflow as tf
import transformers
from concept import f1_metric


def prediction(features, params, mode):
  if 'attr' in params.modules:
    sentence_idxs = features['sentence_idxs']  # [batch_size, num_sentences, sentence_len]
    sentence_lens = features['sentence_lens']  # [batch_size, num_sentences]
    activity_idxs = features['activity_idxs']  # [batch_size, num_sentences, concepts_per_sentence]
    concept_idxs = features['concept_idxs']    # [batch_size, num_sentences, concepts_per_sentence]
    base_encoding = features['base_encoding']  # [batch_size, max_doc_len, emb_size]
    if params.transformer_type == 'adaptive':
      features['sent_ponder_times'], features['sent_remainders'] = [], []

    # [batch_size, num_sentences, sentence_len, emb_size]
    embeddings = batch_gather_nd(tf.expand_dims(sentence_idxs, axis=-1), base_encoding, params.batch_size)

    # [batch_size, num_sentences, sentence_len]
    mask = tf.sequence_mask(sentence_lens,
                            maxlen=params.toks_per_sent,
                            dtype=tf.float32,
                            name='sentence_mask')
    embeddings = embeddings * tf.expand_dims(mask, axis=-1)

    with tf.variable_scope('sentence_transformer', reuse=tf.AUTO_REUSE):
      tparams = transformers.sentence_transformer_params(params)
      sentence_transformer = transformers.create(params, tparams, mode)

      activity_logits = {k: [] for k in params.attr_info.activity_attributes}
      concept_logits = {k: [] for k in params.attr_info.common_attributes}
      activity_idxs = tf.expand_dims(activity_idxs, axis=-1)
      concept_idxs = tf.expand_dims(concept_idxs, axis=-1)
      if params.transformer_type != 'adaptive':
        mask = tf.to_int32(mask)

      for sentence_embs, sentence_mask, aidxs, cidxs in zip(
          tf.unstack(embeddings, axis=1),  # sentence_embs is [batch_size, sentence_len, emb_size]
          tf.unstack(mask, axis=1),  # sentence_mask is [batch_size, sentence_len]
          tf.unstack(activity_idxs, axis=1),  # aidxs is [batch_size, concepts_per_sentence, 1]
          tf.unstack(concept_idxs, axis=1)  # cidxs is [batch_size, concepts_per_sentence, 1]
      ):
        # encoding is [batch_size, sentence_len, emb_size]
        encoding = sentence_transformer.encode_no_lookup(sentence_embs, sentence_mask)
        if params.transformer_type == 'adaptive':
          encoding, (ponder_times, remainders) = encoding
          features['sent_ponder_times'] += [ponder_times]
          features['sent_remainders'] += [remainders]
        attribute_logits(activity_logits,
                         encoding,
                         aidxs,
                         {k: v for k, v in params.attr_info.attr_label_maps.iteritems()
                            if k in params.attr_info.activity_attributes},
                         params.batch_size,
                         mode, params)
        attribute_logits(concept_logits,
                         encoding,
                         cidxs,
                         {k: v for k, v in params.attr_info.attr_label_maps.iteritems()
                            if k in params.attr_info.common_attributes},
                         params.batch_size,
                         mode, params)
    activity_logits = {name: tf.stack(tlist, axis=1) for name, tlist in activity_logits.iteritems()}
    concept_logits = {name: tf.stack(tlist, axis=1) for name, tlist in concept_logits.iteritems()}

    # each logit tensor will be [batch_size, num_sentences, concepts_per_sentence, num_classes]
    logits = dict(activity_logits, **concept_logits)  # suck it guido
    predictions = {name: tf.argmax(l, axis=-1) for name, l in logits.iteritems()}

    # print('Attn graph nodes:')
    # for t in tf.get_default_graph().as_graph_def().node:
    #   if 'attention_weights' in t.name:
    #     print(t.name)

    return logits, predictions
  else:
    return {}, {}


def loss(features, labels, logits, params):
  if 'attr' in params.modules:
    # both masks are [batch_size, num_sentences, concepts_per_sentence]
    activity_mask = tf.sequence_mask(features['activity_lens'], maxlen=params.concepts_per_sent, dtype=tf.float32)
    concept_mask = tf.sequence_mask(features['concept_lens'], maxlen=params.concepts_per_sent, dtype=tf.float32)
    attr_loss = 0
    for attr_name, logs in logits.iteritems():
      mask = concept_mask if attr_name in params.attr_info.common_attributes else activity_mask
      # class weighting
      if params.class_weighting:
        attr_labels = labels[attr_name]
        weights = tf.zeros_like(mask, dtype=tf.float32)
        for _, idx in params.attr_info.attr_label_maps[attr_name].items():
          class_examples = tf.to_float(tf.equal(attr_labels, idx))
          class_weight = params.attr_info.attr_weight_maps[attr_name][idx] - 1.
          weights += (class_weight * class_examples)
        weights += 1.
        weights *= mask
      else:
        weights = mask
      attr_loss += tf.losses.sparse_softmax_cross_entropy(labels[attr_name],
                                                          logs,
                                                          weights=weights,
                                                          scope=attr_name)
    tf.summary.scalar('attribute_loss', attr_loss)
    if params.transformer_type == 'adaptive':
      act_loss = tf.reduce_mean(tf.stack(features['sent_ponder_times']) + tf.stack(features['sent_remainders']))
      tf.summary.scalar('sent_act_loss', act_loss)
      attr_loss += params.act_loss_param * act_loss
    return attr_loss
  else:
    return 0.


def evaluation(features, labels, predictions, params, eval_metric_ops):
  if 'attr' in params.modules:
    attr_map_map = params.attr_info.attr_label_maps
    activity_mask = tf.sequence_mask(features['activity_lens'], maxlen=params.concepts_per_sent, dtype=tf.float32)
    concept_mask = tf.sequence_mask(features['concept_lens'], maxlen=params.concepts_per_sent, dtype=tf.float32)
    for attr_name, pred in predictions.iteritems():
      weights = concept_mask if attr_name in params.attr_info.common_attributes else activity_mask

      class2idx = attr_map_map[attr_name]
      if len(class2idx) == 2:
        # acc = tf.metrics.accuracy(labels[attr_name], pred, weights=weights)
        prec = tf.metrics.accuracy(labels[attr_name], pred, weights=weights)
        rec = tf.metrics.accuracy(labels[attr_name], pred, weights=weights)
        f1 = f1_metric(
          prec_metric=prec,
          recall_metric=rec)
      else:
        tp, tn, fp, fn = 0, 0, 0, 0
        update_ops = []
        f1s = []
        # f1_update_ops, acc_update_ops = [], []
        for attr_class, i in class2idx.items():

          labels_i = tf.equal(labels[attr_name], i)
          predictions_i = tf.equal(pred, i)

          tp_ = tf.metrics.true_positives(labels_i, predictions_i, weights=weights)
          tp += tp_[0]
          tn_ = tf.metrics.true_negatives(labels_i, predictions_i, weights=weights)
          tn += tn_[0]
          fp_ = tf.metrics.false_positives(labels_i, predictions_i, weights=weights)
          fp += fp_[0]
          fn_ = tf.metrics.false_negatives(labels_i, predictions_i, weights=weights)
          fn += fn_[0]
          update_ops += [tp_[1], tn_[1], fp_[1], fn_[1]]

          p = tp_[0] / (tp_[0] + fp_[0] + 1e-5)
          r = tp_[0] / (tp_[0] + fn_[0] + 1e-5)
          f1 = 2 * p * r / (p + r + 1e-5)
          f1s.append(f1)

          if params.verbose_eval:
            eval_metric_ops['%s_%s-recall' % (attr_name, attr_class)] = r, r
            eval_metric_ops['%s_%s-prec' % (attr_name, attr_class)] = p, p
            eval_metric_ops['%s_%s-f1' % (attr_name, attr_class)] = f1, f1
            eval_metric_ops['%s_%s-num' % (attr_name, attr_class)] = tp_[0] + fn_[0], tf.group(tp_[1], fn_[1])
            eval_metric_ops['%s_%s-acc' % (attr_name, attr_class)] = tf.metrics.accuracy(
              labels_i, predictions_i, weights)
        p = tp / (tp + fp + 1e-5)
        r = tp / (tp + fn + 1e-5)
        f1 = 2 * p * r / (p + r + 1e-5), tf.group(update_ops)
        prec = p, tf.group(update_ops)
        rec = r, tf.group(update_ops)
        # acc = (tp + tn) / (tp + tn + fp + fn), tf.group(update_ops)
        macro_f1 = tf.reduce_mean(tf.stack(f1s))
        eval_metric_ops['%s-macro-f1' % attr_name] = macro_f1, tf.group(update_ops)
      # eval_metric_ops['%s-acc' % attr_name] = acc
      eval_metric_ops['%s-f1' % attr_name] = f1
      eval_metric_ops['%s-prec' % attr_name] = prec
      eval_metric_ops['%s-rec' % attr_name] = rec


def attribute_logits(logits_map, encodings, idxs, attr2classes, batch_size, mode, params):
  # [batch_size, concepts_per_sentence, emb_size]
  enc = batch_gather_nd(idxs, encodings, batch_size)

  for attr, logits_list in logits_map.iteritems():
    with tf.variable_scope(attr):
      if params.multi_layer_attr:
        logits = tf.layers.dense(enc, 100, activation=tf.nn.relu)
        if mode == tf.estimator.ModeKeys.TRAIN:
          logits = tf.layers.dropout(logits, rate=params.drop_prob)
      else:
        logits = enc

      logits = tf.squeeze(tf.layers.dense(logits, len(attr2classes[attr])))
      if mode == tf.estimator.ModeKeys.TRAIN:
        logits = tf.layers.dropout(logits, rate=params.drop_prob)
      logits_list.append(logits)


def batch_gather_nd(idxs, params, batch_size):
  # runs gather_nd on each batch slice of idxs and params
  ret = []
  for p, idx in zip(tf.unstack(params, num=batch_size), tf.unstack(idxs, num=batch_size)):
    ret.append(tf.gather_nd(p, idx))
  return tf.stack(ret)


def get_attention_weights():
  # univ
  tensor_name = 'sentence_transformer/encode/encoder_stack_univ/encoder_stack_univ/' \
                'self_attention/self_attention/attention_weights%s:0'

  # trans
  # tensor_name = 'sentence_transformer/encode/encoder_stack/layer_1/' \
  #               'self_attention/self_attention/attention_weights%s:0'
  attention_weights = [tf.get_default_graph().get_tensor_by_name(tensor_name % '')]
  for i in range(1, 25):
    attention_weights.append(tf.get_default_graph().get_tensor_by_name(tensor_name % ('_%d' % i)))
  attention_weights = tf.stack(attention_weights, axis=1, name='all_attention_weights')
  print(attention_weights)
  return attention_weights
