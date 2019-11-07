import manifest from './weights_manifest.json'
import * as tf from '@tensorflow/tfjs-node';
import { WeightsManifestConfig } from '@tensorflow/tfjs-core/dist/io/io';
const CHECKPOINT_URL = 'https://storage.googleapis.com/' +
    'download.magenta.tensorflow.org/models/performance_rnn/tfjs'

export async function loadModel() {
  const weights = await tf.io.loadWeights(manifest as WeightsManifestConfig, CHECKPOINT_URL);
  const models = modelFactory(weights);
  return models
}

export type Models = ReturnType<typeof modelFactory>

function modelFactory(vars: {[varName: string]: tf.Tensor}) {
  const lstmKernel1 =
        vars['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel'] as
        tf.Tensor2D;
  const lstmBias1 = vars['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias'] as
        tf.Tensor1D;
  const lstmKernel2 =
        vars['rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel'] as
        tf.Tensor2D;
  const lstmBias2 = vars['rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias'] as
        tf.Tensor1D;
  const lstmKernel3 =
        vars['rnn/multi_rnn_cell/cell_2/basic_lstm_cell/kernel'] as
        tf.Tensor2D;
  const lstmBias3 = vars['rnn/multi_rnn_cell/cell_2/basic_lstm_cell/bias'] as
        tf.Tensor1D;
  const fcB = vars['fully_connected/biases'] as tf.Tensor1D;
  const fcW = vars['fully_connected/weights'] as tf.Tensor2D;

  return {
    lstmKernel1,
    lstmBias1,
    lstmKernel2,
    lstmBias2,
    lstmKernel3,
    lstmBias3,
    fcB,
    fcW
  }
}