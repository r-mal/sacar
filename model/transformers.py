import tensorflow as tf
from official.transformer.model import attention_layer, ffn_layer, transformer, model_utils, model_params
from tensor2tensor.models.research import universal_transformer, universal_transformer_util
from tensor2tensor.models import transformer as t2t_transformer


def create(params, transformer_params, mode):
  transformer_params.use_convolution = params.use_convolution
  constructor = {'transformer': lambda: Transformer(transformer_params, mode == tf.estimator.ModeKeys.TRAIN),
                 'universal': lambda: UniversalTransformer(transformer_params, mode == tf.estimator.ModeKeys.TRAIN),
                 'adaptive': lambda: AdaptiveUniversalTransformer(transformer_params, mode)}
  return constructor[params.transformer_type]()


def sentence_transformer_params(params):
  if params.transformer_type == 'adaptive':
    hparams = universal_transformer.adaptive_universal_transformer_base()
    hparams.batch_size = params.batch_size
    hparams.recurrence_type = 'act'
    hparams.act_type = 'basic'
    hparams.max_length = params.toks_per_sent
    hparams.num_hidden_layers = params.sentence_transformer_num_stacks
    hparams.add_hparam("vocab_size", params.vocab_size)
    hparams.hidden_size = params.embedding_size
    hparams.num_heads = params.sentence_transformer_num_heads
    hparams.add_hparam("embedding_device", params.embedding_device)
    return hparams
  tparams = model_params.TransformerBaseParams()
  tparams.batch_size = params.batch_size
  tparams.max_length = params.toks_per_sent
  tparams.vocab_size = params.vocab_size
  tparams.hidden_size = params.embedding_size
  tparams.num_heads = params.sentence_transformer_num_heads
  tparams.num_hidden_layers = params.sentence_transformer_num_stacks
  return tparams


def base_transformer_params(params):
  if params.transformer_type == 'adaptive':
    hparams = universal_transformer.adaptive_universal_transformer_base()
    hparams.batch_size = params.batch_size
    hparams.recurrence_type = 'act'
    hparams.act_type = 'basic'
    hparams.max_length = params.max_doc_len
    hparams.num_hidden_layers = params.base_transformer_num_stacks
    hparams.add_hparam("vocab_size", params.vocab_size)
    hparams.hidden_size = params.embedding_size
    hparams.num_heads = params.base_transformer_num_heads
    hparams.add_hparam("embedding_device", params.embedding_device)
    return hparams
  tparams = model_params.TransformerBaseParams()
  tparams.batch_size = params.batch_size
  tparams.max_length = params.max_doc_len
  tparams.vocab_size = params.vocab_size
  tparams.hidden_size = params.embedding_size
  tparams.num_heads = params.base_transformer_num_heads
  tparams.num_hidden_layers = params.base_transformer_num_stacks
  return tparams


class Transformer(transformer.Transformer):
  def encode_no_lookup(self, embedded_inputs, inputs_mask):
    """Encoder step for transformer given already-embedded inputs

      Args:
        model: transformer model
        embedded_inputs: int tensor with shape [batch_size, input_length, emb_size].
        inputs_mask: int tensor with shape [batch_size, input_length]
        params: transformer_params
        train: boolean flag

      Returns:
        float tensor with shape [batch_size, input_length, hidden_size]
      """
    with tf.name_scope("encode"):
      # Prepare inputs to the layer stack by adding positional encodings and
      # applying dropout.
      inputs_padding = model_utils.get_padding(inputs_mask)
      attention_bias = model_utils.get_padding_bias(inputs_mask)

      with tf.name_scope("add_pos_encoding"):
        length = tf.shape(embedded_inputs)[1]
        pos_encoding = model_utils.get_position_encoding(
          length, self.params.hidden_size)
        encoder_inputs = embedded_inputs + pos_encoding

      if self.train:
        encoder_inputs = tf.nn.dropout(
          encoder_inputs, 1 - self.params.layer_postprocess_dropout)

      return self.encoder_stack(encoder_inputs, attention_bias, inputs_padding)


class UniversalTransformer(Transformer):
  def __init__(self, params, train):
    super(UniversalTransformer, self).__init__(params, train)
    self.encoder_stack = EncoderStackUniv(params, train)


class AdaptiveUniversalTransformer:
  def __init__(self, params, mode):
    params.filter_size = 512
    self.hparams = params
    self.target_space = 0
    self.model = universal_transformer.UniversalTransformerEncoder(params, mode)
    self.embedding_device = params.embedding_device
    with tf.device(self.embedding_device):
      self.word_embeddings = tf.get_variable("word_embeddings", shape=[params.vocab_size, params.hidden_size])

  def encode(self, inputs, input_mask=None, _target_space=None, _hparams=None, _features=None, _losses=None):
    """Encode transformer inputs.

    Args:
      inputs: Transformer inputs [batch_size, input_length] int64 tensor.
      input_mask: [batch_size, input_length, hidden_size] mask
      _target_space: scalar, target space ID.
      _hparams: hyperparmeters for model.
      _features: optionally pass the entire features dictionary as well.
        This is needed now for "packed" datasets.
      _losses: Unused.

    Returns:
      Tuple of:
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_extra_output: which is extra encoder output used in some
            variants of the model (e.g. in ACT, to pass the ponder-time to body)
    """
    # [batch_size, input_length]
    nonpadding = tf.to_float(tf.not_equal(inputs, 0))

    with tf.device(self.embedding_device):
      # [batch_size, input_length, hidden_dim]
      inputs = tf.nn.embedding_lookup(self.word_embeddings, inputs)
    if input_mask is not None:
      inputs += input_mask

    return self.encode_no_lookup(inputs, nonpadding)

  def encode_no_lookup(self, embedded_inputs, inputs_mask):
    """Encoder step for transformer given already-embedded inputs

      Args:
        embedded_inputs: int tensor with shape [batch_size, input_length, emb_size].
        inputs_mask: tensor with shape [batch_size, input_length]

      Returns:
        float tensor with shape [batch_size, input_length, hidden_size]
      """
    (encoder_input, self_attention_bias, _) = (
      t2t_transformer.transformer_prepare_encoder(embedded_inputs, self.target_space, self.hparams))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - self.hparams.layer_prepostprocess_dropout)

    (encoder_output, encoder_extra_output) = (
      universal_transformer_util.universal_transformer_encoder(
        encoder_input,
        self_attention_bias,
        self.hparams,
        nonpadding=inputs_mask,
        save_weights_to=self.model.attention_weights))

    return encoder_output, encoder_extra_output


class EncoderStack(tf.layers.Layer):
  """Transformer encoder stack.

  The encoder stack is made up of N identical layers. Each layer is composed
  of the sublayers:
    1. Self-attention layer
    2. Feedforward network (which is 2 fully-connected layers)
  """

  def __init__(self, params, train):
    super(EncoderStack, self).__init__()
    self.layers = []
    for _ in range(params.num_hidden_layers):
      # Create sublayers for each layer.
      self_attention_layer = attention_layer.SelfAttention(
          params.hidden_size, params.num_heads, params.attention_dropout, train)
      feed_forward_network = ConvolutionalFeedForwardNetwork(
          params.hidden_size, params.filter_size, params.relu_dropout, train) if params.use_convolution else \
          ffn_layer.FeedFowardNetwork(params.hidden_size, params.filter_size, params.relu_dropout, train)

      self.layers.append([
        transformer.PrePostProcessingWrapper(self_attention_layer, params, train),
        transformer.PrePostProcessingWrapper(feed_forward_network, params, train)])

    # Create final layer normalization layer.
    self.output_normalization = transformer.LayerNormalization(params.hidden_size)


# noinspection PyAbstractClass
class EncoderStackUniv(tf.layers.Layer):
  """Transformer encoder stack.

  The encoder stack is made up of N identical layers. Each layer is composed
  of the sublayers:
    1. Self-attention layer
    2. Feedforward network (which is 2 fully-connected layers)
  """

  def __init__(self, params, train):
    super(EncoderStackUniv, self).__init__()
    self.layers = []
    for _ in range(params.num_hidden_layers):
      # Create sublayers for each layer.
      self_attention_layer = attention_layer.SelfAttention(
          params.hidden_size, params.num_heads, params.attention_dropout, train)
      feed_forward_network = ConvolutionalFeedForwardNetwork(
          params.hidden_size, params.filter_size, params.relu_dropout, train) if params.use_convolution else \
          ffn_layer.FeedFowardNetwork(params.hidden_size, params.filter_size, params.relu_dropout, train)

      self.layers.append([
          PrePostProcessingWrapper(self_attention_layer, params, train),
          PrePostProcessingWrapper(feed_forward_network, params, train)])

    # Create final layer normalization layer.
    self.output_normalization = transformer.LayerNormalization(params.hidden_size)

  # noinspection PyMethodOverriding
  def call(self, encoder_inputs, attention_bias, inputs_padding):
    """Return the output of the encoder layer stacks.

    Args:
      encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
      attention_bias: bias for the encoder self-attention layer.
        [batch_size, 1, 1, input_length]
      inputs_padding: P

    Returns:
      Output of encoder layer stack.
      float32 tensor with shape [batch_size, input_length, hidden_size]
    """
    for n, layer in enumerate(self.layers):
      # Run inputs through the sublayers.
      self_attention_layer = layer[0]
      feed_forward_network = layer[1]

      with tf.variable_scope("encoder_stack_univ", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("self_attention"):
          encoder_inputs = self_attention_layer(encoder_inputs, attention_bias)
        with tf.variable_scope("ffn"):
          encoder_inputs = feed_forward_network(encoder_inputs, inputs_padding)

    return self.output_normalization(encoder_inputs)


class PrePostProcessingWrapper(object):
  """Wrapper class that applies layer pre-processing and post-processing.

  Applies the passed layer, a residual connection,  dropout, then layer normalization.
  """

  def __init__(self, layer, params, train):
    self.layer = layer
    self.postprocess_dropout = params.layer_postprocess_dropout
    self.train = train

    # Create normalization layer
    self.layer_norm = transformer.LayerNormalization(params.hidden_size)

  def __call__(self, x, *args, **kwargs):
    # Get layer output
    y = self.layer(x, *args, **kwargs)

    # residual connection
    y = y + x

    # Postprocessing: apply dropout and residual connection
    if self.train:
      global y
      y = tf.nn.dropout(y, 1 - self.postprocess_dropout)

    # Preprocessing: apply layer normalization
    y = self.layer_norm(x)

    return y


class ConvolutionalFeedForwardNetwork(tf.layers.Layer):
  """Fully connected feedforward network."""

  def __init__(self, hidden_size, filter_size, relu_dropout, train):
    super(ConvolutionalFeedForwardNetwork, self).__init__()
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.relu_dropout = relu_dropout
    self.train = train

    self.filter_dense_layer = tf.layers.Dense(
        filter_size, use_bias=True, activation=tf.nn.relu, name="filter_layer")
    self.convolutional_layer = tf.layers.Conv1D(
        filters=filter_size, kernel_size=5, use_bias=True, activation=tf.nn.relu, name='conv_layer')
    self.output_dense_layer = tf.layers.Dense(
        hidden_size, use_bias=True, name="output_layer")

  def call(self, x, padding=None):
    """Return outputs of the feedforward network.

    Args:
      x: tensor with shape [batch_size, length, hidden_size]
      padding: (optional) If set, the padding values are temporarily removed
        from x. The padding values are placed back in the output tensor in the
        same locations. shape [batch_size, length]

    Returns:
      Output of the feedforward network.
      tensor with shape [batch_size, length, hidden_size]
    """
    # Retrieve dynamically known shapes
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]

    if padding is not None:
      with tf.name_scope("remove_padding"):
        # Flatten padding to [batch_size*length]
        pad_mask = tf.reshape(padding, [-1])

        nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))

        # Reshape x to [batch_size*length, hidden_size] to remove padding
        x = tf.reshape(x, [-1, self.hidden_size])
        x = tf.gather_nd(x, indices=nonpad_ids)

        # Reshape x from 2 dimensions to 3 dimensions.
        x.set_shape([None, self.hidden_size])
        x = tf.expand_dims(x, axis=0)

    output = self.filter_dense_layer(x)
    if self.train:
      output = tf.nn.dropout(output, 1.0 - self.relu_dropout)
    output = self.convolutional_layer(output)
    output = self.output_dense_layer(output)

    if padding is not None:
      with tf.name_scope("re_add_padding"):
        output = tf.squeeze(output, axis=0)
        output = tf.scatter_nd(
            indices=nonpad_ids,
            updates=output,
            shape=[batch_size * length, self.hidden_size]
        )
        output = tf.reshape(output, [batch_size, length, self.hidden_size])
    return output
