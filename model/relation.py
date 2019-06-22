import tensorflow as tf

from attribute import batch_gather_nd


def prediction(features, params, mode):
  if 'relation' in params.modules:
    # concept_mentions refers to the number of concept mentions in the document
    # global_concepts refers to the number of unique concepts in the document
    base_encoding = features['base_encoding']  # [batch_size, max_doc_len, emb_size]
    concept_idxs = features['concept_mention_idxs']  # [batch_size, concept_mentions]
    # [batch_size, global_concepts, global_concepts, mention_pairs, 2] last dim is [mention, mention] index list into
    # affinity_scores
    mentions_per_concept = features['mentions_per_concept']
    mention_pairs_per_conc_pair = features['mention_pairs_per_conc_pair']  # [bath_size, global_concepts, global_concepts]
    num_relations = len(params.rel2id)

    # [batch_size, concept_mentions, emb_size]
    base_encoding = batch_gather_nd(tf.expand_dims(concept_idxs, axis=-1), base_encoding, params.batch_size)
    print(base_encoding)

    # both are [batch_size, concept_mentions, emb_size]
    with tf.variable_scope('source'):
      source_encoding = argument_encoding(base_encoding, params, mode)
    with tf.variable_scope('destination'):
      dest_encoding = argument_encoding(base_encoding, params, mode)

    # [num_rels, emb_size, emb_size]
    relation_embeddings = tf.get_variable('rel_embs', [num_relations, params.rel_hidden_size, params.rel_hidden_size])
    # [batch_size, num_rels, emb_size, emb_size]
    relation_embeddings = tf.tile(tf.expand_dims(relation_embeddings, axis=0), [params.batch_size, 1, 1, 1])
    # [batch_size, num_rels, concept_mentions, emb_size]
    source_encoding = tf.tile(tf.expand_dims(source_encoding, axis=1), [1, num_relations, 1, 1])
    # [batch_size, num_rels, emb_size, concept_mentions]
    dest_encoding = tf.tile(tf.expand_dims(tf.transpose(dest_encoding, [0, 2, 1]),
                                           axis=1),
                            [1, num_relations, 1, 1])
    # [batch_size, num_rels, concept_mentions, concept_mentions]
    affinity_scores = tf.matmul(tf.matmul(source_encoding, relation_embeddings), dest_encoding)
    # [batch_size, concept_mentions, concept_mentions, num_rels]
    affinity_scores = tf.transpose(affinity_scores, [0, 2, 3, 1])

    # [batch_size, global_concepts, global_concepts, num_pairs, num_rels]
    score_pairs = batch_gather_nd(mentions_per_concept, affinity_scores, params.batch_size)
    score_pairs *= tf.expand_dims(tf.sequence_mask(mention_pairs_per_conc_pair,
                                                   maxlen=params.mention_pairs,
                                                   dtype=tf.float32),
                                  axis=-1)
    # [batch_size, global_concepts, global_concepts, num_rels]
    scores = tf.log(tf.reduce_sum(tf.exp(score_pairs), axis=3) + 1e-7, name='relation_scores')
    print(scores)
    predictions = tf.argmax(scores, axis=-1)
    return scores, predictions
  else:
    return None, None


def loss(features, labels, logits, params):
  relation_loss = 0.
  if 'relation' in params.modules:
    relation_mask = tf.to_float(features['relation_mask'])
    # weight non-NONE relations 1000x higher than NONE relations
    non_zero_weights = tf.to_float(tf.not_equal(labels['relation_labels'], 0)) * 100.
    relation_mask = relation_mask * (relation_mask + non_zero_weights) * 1e-2
    print(labels['relation_labels'])
    print(features['relation_mask'])
    print(logits)
    relation_loss = tf.losses.sparse_softmax_cross_entropy(labels['relation_labels'],
                                                           logits,
                                                           weights=relation_mask)
    tf.summary.scalar('relation_loss', relation_loss)
  return relation_loss


def evaluation(features, labels, predictions, params, eval_metric_ops):
  if 'relation' in params.modules:
    relation_mask = features['relation_mask']
    for name, i in params.rel2id.iteritems():
      labels_i = tf.equal(labels['relation_labels'], i)
      predictions_i = tf.equal(predictions, i)
      eval_metric_ops['%s-recall' % name] = tf.metrics.recall(labels_i, predictions_i, weights=relation_mask)
      eval_metric_ops['%s-prec' % name] = tf.metrics.precision(labels_i, predictions_i, weights=relation_mask)
      eval_metric_ops['%s-acc' % name] = tf.metrics.accuracy(labels_i, predictions_i, weights=relation_mask)
      if params.verbose_eval:
        eval_metric_ops['%s-tp' % name] = tp = tf.metrics.true_positives(labels_i, predictions_i, weights=relation_mask)
        eval_metric_ops['%s-fp' % name] = tf.metrics.false_positives(labels_i, predictions_i, weights=relation_mask)
        eval_metric_ops['%s-fn' % name] = fn = tf.metrics.false_negatives(labels_i, predictions_i, weights=relation_mask)
        eval_metric_ops['%s-num' % name] = tp[0] + fn[0], tf.group(tp[1], fn[1])


def argument_encoding(base_encoding, params, mode):
    enc = tf.layers.dense(tf.layers.dense(base_encoding,
                                          params.rel_hidden_size,
                                          activation=tf.nn.relu,
                                          use_bias=False),
                          params.rel_hidden_size)
    if mode == tf.estimator.ModeKeys.TRAIN:
      enc = tf.layers.dropout(enc, params.drop_prob)
    return enc
