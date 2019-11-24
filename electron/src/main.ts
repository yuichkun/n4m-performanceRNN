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
import { Models } from './modelLoader'
import { EVENT_SIZE, EVENT_RANGES, MIDI_EVENT_ON, MIDI_EVENT_OFF, VELOCITY_BINS } from './constants'
// import { performance } from 'perf_hooks' // TODO: make this isomorphic
import { IMidiGateway } from './MidiGateway'

export function getInitialWeights(models: Models): tf.Tensor<tf.Rank.R2>[]{
  const { lstmBias1, lstmBias2, lstmBias3 } = models
  if(!(lstmBias1 && lstmBias2 && lstmBias3)) throw new TypeError('Invalid Model shape. Model must have the following properties. lstmBias1, lstmBias2, lstmBias3')
  return [
    tf.zeros([1, lstmBias1.shape[0] / 4]),
    tf.zeros([1, lstmBias2.shape[0] / 4]),
    tf.zeros([1, lstmBias3.shape[0] / 4]),
  ];
}

interface ConstructorOptions {
  models: Models
  midiGateway: IMidiGateway
}
// TODO: add logger
export default class PerformanceRnn {
  cellState: tf.Tensor2D[] 
  hiddenState: tf.Tensor2D[] 
  lastSample: tf.Tensor<tf.Rank.R0> | null
  currentLoopId: number
  currentTimeSec: number
  conditioned: boolean
  noteDensityEncoding: tf.Tensor1D | null;
  pitchHistogramEncoding: tf.Tensor1D | null;
  models: Models
  activeNotes = new Map<number, number>();
  midiGateway: IMidiGateway

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
  static DENSITY_BIN_RANGES = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];

  constructor(options: ConstructorOptions) {
    const { models, midiGateway } = options
    this.cellState = getInitialWeights(models)
    this.hiddenState = getInitialWeights(models)
    this.lastSample = tf.scalar(PerformanceRnn.PRIMER_IDX, 'int32');
    this.currentLoopId = 0
    this.currentTimeSec = 0
    this.conditioned = false
    this.noteDensityEncoding = null
    this.pitchHistogramEncoding = null
    this.models = models
    this.midiGateway = midiGateway
  }

  async initialize() {
    this.currentLoopId = 0
    this.updateConditioningParams()
  }
  
  resetRnn() {
    this.cellState = getInitialWeights(this.models)
    this.hiddenState = getInitialWeights(this.models)
    if (this.lastSample != null) {
      this.lastSample.dispose();
    }
    this.lastSample = tf.scalar(PerformanceRnn.PRIMER_IDX, 'int32');

    // TODO replace this
    // currentTimeSec = piano.now();
    // pianoStartTimestampMs = performance.now() - currentTimeSec * 1000;

    // TODO: necessary?
    this.generateStep(this.currentLoopId);
  }

  async generateStep(loopId: number) {
    
    const { lstmKernel1, lstmKernel2, lstmKernel3, lstmBias1, lstmBias2, lstmBias3 } = this.models
    if (
      !(
        lstmKernel1 &&
        lstmKernel2 &&
        lstmKernel3 &&
        lstmBias1 &&
        lstmBias2 &&
        lstmBias3
      )
    ) throw new TypeError('Model has invalid shape')

    const lstm1 = (data: tf.Tensor2D, c: tf.Tensor2D, h: tf.Tensor2D) =>
        tf.basicLSTMCell(PerformanceRnn.forgetBias, lstmKernel1, lstmBias1, data, c, h);
    const lstm2 = (data: tf.Tensor2D, c: tf.Tensor2D, h: tf.Tensor2D) =>
        tf.basicLSTMCell(PerformanceRnn.forgetBias, lstmKernel2, lstmBias2, data, c, h);
    const lstm3 = (data: tf.Tensor2D, c: tf.Tensor2D, h: tf.Tensor2D) =>
        tf.basicLSTMCell(PerformanceRnn.forgetBias, lstmKernel3, lstmBias3, data, c, h);

    // @ts-ignore
    const [_, __, outputs] = tf.tidy(() => {
      // Generate some notes.
      const innerOuts: tf.Scalar[] = [];
      for (let i = 0; i < PerformanceRnn.STEPS_PER_GENERATE_CALL; i++) {
        // Use last sampled output as the next input.
        const eventInput = tf.oneHot(
          this.lastSample!.as1D(), EVENT_SIZE).as1D(); // TODO: better not use !, rather stop using null
        // Dispose the last sample from the previous generate call, 
        // since we kept it.
        if (i === 0 && this.lastSample) {
          this.lastSample.dispose();
          this.lastSample = null
        }
        const conditioning = this.getConditioning();
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
    // if (piano.now() - currentTimeSec > MAX_GENERATION_LAG_SECONDS) {
    //   console.warn(
    //       `Generation is ${piano.now() - currentTimeSec} seconds behind, ` +
    //       `which is over ${MAX_NOTE_DURATION_SECONDS}. Resetting time!`);
    //   currentTimeSec = piano.now();
    // }


    // const delta = Math.max(
    //     0, this.currentTimeSec - piano.now() - PerformanceRnn.GENERATION_BUFFER_SECONDS);
    // setTimeout(() => this.generateStep(loopId), delta * 1000);

    // TODO: is this right?
    setTimeout(() => this.generateStep(loopId), PerformanceRnn.GENERATION_BUFFER_SECONDS);
  }

  playOutput(index: number) {
    let offset = 0;
    for (const eventRange of EVENT_RANGES) {
      const [eventType, minValue, maxValue] = eventRange
      if (offset <= index && index <= offset + maxValue - minValue) {
        switch (eventType) {
          case 'note_on': {
            const noteNum = index - offset;

            this.activeNotes.set(noteNum, this.currentTimeSec);
            this.midiGateway.activeDevice.send(
                [
                  MIDI_EVENT_ON, noteNum,
                  Math.min(this.midiGateway.currentVelocity, 127)
                ])
            break;
          }
          case 'note_off': {
            const noteNum = index - offset;
    
            const activeNoteEndTimeSec = this.activeNotes.get(noteNum);
            // If the note off event is generated for a note that hasn't been
            // pressed, just ignore it.
            if (activeNoteEndTimeSec == null) break;

            // TODO this maybe necessary. idk.
            // const timeSec =
            //     Math.max(this.currentTimeSec, activeNoteEndTimeSec + .5);
    
            this.midiGateway.activeDevice.send(
                [
                  MIDI_EVENT_OFF, noteNum,
                  Math.min(this.midiGateway.currentVelocity, 127)
                ])
            this.activeNotes.delete(noteNum);
            break;
          }
          case 'time_shift': {
            const STEPS_PER_SECOND = 100;
            const MAX_NOTE_DURATION_SECONDS = 3;
            this.currentTimeSec += (index - offset + 1) / STEPS_PER_SECOND;
            this.activeNotes.forEach((timeSec, noteNum) => {
              if (this.currentTimeSec - timeSec > MAX_NOTE_DURATION_SECONDS) {
                // TODO: fix log here
                console.info(
                    `Note ${noteNum} has been active for ${
                        this.currentTimeSec - timeSec}, ` +
                    `seconds which is over ${MAX_NOTE_DURATION_SECONDS}, will ` +
                    `release.`);
                this.midiGateway.activeDevice.send([
                  MIDI_EVENT_OFF, noteNum,
                  Math.min(this.midiGateway.currentVelocity, 127)
                ]);
                this.activeNotes.delete(noteNum);
              }
            });
            break;
          }
          case 'velocity_change': {
            const nextVelocity = ((index - offset + 1) * Math.ceil(127 / VELOCITY_BINS)) / 127;
            this.midiGateway.changeVelocity(nextVelocity)
            break;
          }
          default: {
            throw new Error(`Unknown event type: ${eventType}`)
          }
        }
      }
      offset += maxValue - minValue + 1;
    }
  }

  getConditioning(): tf.Tensor1D {
    return tf.tidy(() => {
      const { conditioned, noteDensityEncoding, pitchHistogramEncoding } = this;
      if(!(noteDensityEncoding && pitchHistogramEncoding)) throw new TypeError('noteDensityEncoding and pitchHistogramEncoding must be defined')
      if (!conditioned) {
        const size = 1 + (noteDensityEncoding.shape[0] as number) +
            (pitchHistogramEncoding.shape[0] as number);
        const conditioning: tf.Tensor1D =
            tf.oneHot(tf.tensor1d([0], 'int32'), size).as1D();
        return conditioning;
      } else {
        const axis = 0;
        const conditioningValues =
            noteDensityEncoding.concat(pitchHistogramEncoding, axis);
        return tf.tensor1d([0], 'int32').concat(conditioningValues, axis);
      }
    });
  }

  updateConditioningParams() {
    const pitchHistogram = Array(12).fill(0) // TODO: make this controllable

    if (this.noteDensityEncoding != null) {
      this.noteDensityEncoding.dispose();
      this.noteDensityEncoding = null;
    }

    const noteDensityIdx = 0; // TODO: make this controllable

    this.noteDensityEncoding =
        tf.oneHot(
            tf.tensor1d([noteDensityIdx + 1], 'int32'),
            PerformanceRnn.DENSITY_BIN_RANGES.length + 1).as1D();

    if (this.pitchHistogramEncoding != null) {
      this.pitchHistogramEncoding.dispose();
      this.pitchHistogramEncoding = null;
    }

    const PITCH_HISTOGRAM_SIZE = 12;
    const buffer = tf.buffer<tf.Rank.R1>([PITCH_HISTOGRAM_SIZE], 'float32');
    const pitchHistogramTotal = pitchHistogram.reduce((prev, val) => {
      return prev + val;
    });
    for (let i = 0; i < PITCH_HISTOGRAM_SIZE; i++) {
      buffer.set(pitchHistogram[i] / pitchHistogramTotal, i);
    }
    this.pitchHistogramEncoding = buffer.toTensor();
  }
}