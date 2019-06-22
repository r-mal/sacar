import tensorflow as tf
import os

from data import dataset as ds, pretrain as ptds
from model import Model, pretrain as pt_model

tf.logging.set_verbosity(tf.logging.INFO)


def main(config):
  model_dir = os.path.join(config.model_dir, config.run_name)

  if config.pretrained and not tf.train.checkpoint_exists(model_dir):
    print('Loading pretrained base transformer...')
    exit()
    tf.train.init_from_checkpoint(config.pretrain_data_dir, {'base_transformer/': 'base_transformer/'})

  run_config = tf.estimator.RunConfig(
    model_dir=model_dir,
    save_checkpoints_steps=100,
    save_summary_steps=10,
    log_step_count_steps=10
  )
  train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: ds.load(config, os.path.join(config.record_dir, config.train_filename)).repeat(),
    max_steps=config.steps_per_epoch*config.num_epochs
  )
  print("Training on %d minibatches" % (config.steps_per_epoch * config.num_epochs))
  eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda: ds.load(config, os.path.join(config.record_dir, config.val_filename)),
    steps=None,
    name='validation',
    start_delay_secs=config.eval_delay,
    throttle_secs=config.eval_throttle
  )

  estimator = tf.estimator.Estimator(
    model_fn=Model.model_fn,
    config=run_config,
    params=config,
    warm_start_from=config.warm_start_model
  )

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def pretrain(config):
  model_dir = os.path.join(config.pretrain_data_dir, 'model', config.run_name)
  run_config = tf.estimator.RunConfig(
    model_dir=model_dir,
    save_checkpoints_steps=1000,
    save_summary_steps=100,
    log_step_count_steps=100
  )
  train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: ptds.load(config, os.path.join(config.pretrain_data_dir, config.train_filename)).repeat()
  )
  print("Training on %d minibatches" % (config.steps_per_epoch * config.num_epochs))
  eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda: ptds.load(config, os.path.join(config.pretrain_data_dir, config.val_filename)),
    steps=None,
    name='validation',
    start_delay_secs=config.eval_delay,
    throttle_secs=config.eval_throttle
  )

  estimator = tf.estimator.Estimator(
    model_fn=pt_model.model_fn,
    config=run_config,
    params=config
  )

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
