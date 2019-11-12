/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as tf from '@tensorflow/tfjs-node'
import {loadModel, Models} from './modelLoader'
import { EVENT_SIZE } from './constants'

function getInitialWeights(models: Models): tf.Tensor<tf.Rank.R2>[]{
  const { lstmBias1, lstmBias2, lstmBias3 } = models
  return [
    tf.zeros([1, lstmBias1.shape[0] / 4]),
    tf.zeros([1, lstmBias2.shape[0] / 4]),
    tf.zeros([1, lstmBias3.shape[0] / 4]),
  ];
}

// TODO: add logger
class PerformanceRnn {
  cellState: tf.Tensor2D[] 
  hiddenState: tf.Tensor2D[] 
  lastSample: tf.Tensor<tf.Rank.R0> | null
  currentLoopId: number
  models: Models
  currentPianoTimeSec: number

  /** @description
   * How many steps to generate per generateStep call.
   * Generating more steps makes it less likely that we'll lag behind in note
   * generation. Generating fewer steps makes it less likely that the browser UI
   * thread will be starved for cycles.
  */
  static STEPS_PER_GENERATE_CALL = 10;
  static forgetBias = tf.scalar(1.0)
  /**
   * shift 1s
   */
  static PRIMER_IDX = 355; 
  /** @description
   * How much time to try to generate ahead. More time means fewer buffer
   * underruns, but also makes the lag from UI change to output larger.
   */
  static GENERATION_BUFFER_SECONDS = .5;

  constructor() {
    this.initialize()
  }

  async initialize() {
    try {
      this.models = await loadModel()
      this.currentLoopId = 0
    } catch (e) {
      console.error("Failed Initializing")
      throw e
    }
  }
  
  resetRnn() {
    this.cellState = getInitialWeights(this.models)
    this.hiddenState = getInitialWeights(this.models)
    if (this.lastSample != null) {
      this.lastSample.dispose();
    }
    this.lastSample = tf.scalar(PerformanceRnn.PRIMER_IDX, 'int32');

    // TODO replace this
    // currentPianoTimeSec = piano.now();
    // pianoStartTimestampMs = performance.now() - currentPianoTimeSec * 1000;

    // TODO: necessary?
    this.generateStep(this.currentLoopId);
  }

  async generateStep(loopId: number) {
    
    const { lstmKernel1, lstmKernel2, lstmKernel3, lstmBias1, lstmBias2, lstmBias3 } = this.models

    const lstm1 = (data: tf.Tensor2D, c: tf.Tensor2D, h: tf.Tensor2D) =>
        tf.basicLSTMCell(PerformanceRnn.forgetBias, lstmKernel1, lstmBias1, data, c, h);
    const lstm2 = (data: tf.Tensor2D, c: tf.Tensor2D, h: tf.Tensor2D) =>
        tf.basicLSTMCell(PerformanceRnn.forgetBias, lstmKernel2, lstmBias2, data, c, h);
    const lstm3 = (data: tf.Tensor2D, c: tf.Tensor2D, h: tf.Tensor2D) =>
        tf.basicLSTMCell(PerformanceRnn.forgetBias, lstmKernel3, lstmBias3, data, c, h);

    const [c, h, outputs] = tf.tidy(() => {
      // Generate some notes.
      const innerOuts: tf.Scalar[] = [];
      for (let i = 0; i < PerformanceRnn.STEPS_PER_GENERATE_CALL; i++) {
        // Use last sampled output as the next input.
        const eventInput = tf.oneHot(
          this.lastSample.as1D(), EVENT_SIZE).as1D();
        // Dispose the last sample from the previous generate call, 
        // since we kept it.
        if (i === 0) {
          this.lastSample.dispose();
        }

        // TODO: get conditioning
        // const conditioning = getConditioning();
        const axis = 0;
        const input = conditioning.concat(eventInput, axis).toFloat();
        const output =
            tf.multiRNNCell([lstm1, lstm2, lstm3], input.as2D(1, -1), this.cellState, this.hiddenState);
        this.cellState.forEach(c => c.dispose());
        this.hiddenState.forEach(h => h.dispose());
        this.cellState = output[0];
        this.hiddenState = output[1];

        const outputH = this.hiddenState[2];
        const logits = outputH.matMul(this.models.fcW).add(this.models.fcB);

        const sampledOutput = tf.multinomial(logits.as1D(), 1).asScalar();

        innerOuts.push(sampledOutput);
        this.lastSample = sampledOutput;
      }
      return [this.cellState, this.hiddenState, innerOuts]
    });

    for (let i = 0; i < outputs.length; i++) {
      this.playOutput(outputs[i].dataSync()[0]);
    }


    // TODO: warn here
    // if (piano.now() - currentPianoTimeSec > MAX_GENERATION_LAG_SECONDS) {
    //   console.warn(
    //       `Generation is ${piano.now() - currentPianoTimeSec} seconds behind, ` +
    //       `which is over ${MAX_NOTE_DURATION_SECONDS}. Resetting time!`);
    //   currentPianoTimeSec = piano.now();
    // }


    // const delta = Math.max(
    //     0, this.currentPianoTimeSec - piano.now() - PerformanceRnn.GENERATION_BUFFER_SECONDS);
    // setTimeout(() => this.generateStep(loopId), delta * 1000);

    // TODO: is this right?
    setTimeout(() => this.generateStep(loopId), PerformanceRnn.GENERATION_BUFFER_SECONDS);
  }

  // TODO: implement this
  playOutput(index: number){}
}


