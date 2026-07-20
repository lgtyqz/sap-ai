// network.js
//
// This file builds the neural networks at the heart of Stochastic MuZero.
// If you've read about the algorithm, you'll know it's made of a handful
// of small functions that pass information between each other. Here's a
// quick recap of each one, and what it does in this file:
//
//   h  (representation)       real observation      -> hidden state
//   f  (prediction)           hidden state          -> policy + value
//   phi (afterstateDynamics)  hidden state + action -> afterstate
//   psi (afterstatePrediction) afterstate           -> chance-outcome
//                                                       distribution + Q-value
//   g  (dynamics)             afterstate + chance   -> next hidden state
//                             outcome                  + reward
//   e  (encoder)              real FUTURE observation -> chance-outcome code
//                                                        (training only)
//
// The "afterstate" is the clever bit Stochastic MuZero adds on top of
// MuZero: it's the imagined situation right after you've acted, but
// before randomness (a dice roll, a random tile spawn, etc.) has been
// resolved. Splitting the transition into "you act" -> afterstate ->
// "chance resolves" -> next state is what lets the model represent
// randomness at all, instead of being forced to predict one single
// (deterministic) outcome for every action.
//
// All five functions are small multi-layer perceptrons (MLPs) with a
// couple of residual blocks (a "shortcut connection" pattern that makes
// deeper networks easier to train). They all operate on BATCHES of
// examples at once (shape [batchSize, ...]) since that's what's efficient
// for gradient-based training. Elsewhere in the codebase (mcts.js) we
// use "single-example" convenience wrappers defined at the bottom of this
// file, since tree search naturally works one imagined position at a time.

const tf = require('@tensorflow/tfjs');

/**
 * Builds a small residual MLP: an input layer, followed by a number of
 * "residual blocks" (two dense layers with a shortcut connection added
 * back in, which helps gradients flow through deeper networks), followed
 * by one or more named output heads.
 *
 * @param {number} inputDim - size of the input vector
 * @param {Object} outputHeads - map of headName -> { size, activation }
 * @param {number} hiddenSize
 * @param {number} numResidualBlocks
 * @param {string} namePrefix - used to keep layer names unique across the
 *   five different networks we build below
 */
function buildResidualMLP(inputDim, outputHeads, hiddenSize, numResidualBlocks, namePrefix) {
  const input = tf.input({ shape: [inputDim] });

  let x = tf.layers
    .dense({ units: hiddenSize, activation: 'relu', name: `${namePrefix}_in` })
    .apply(input);

  for (let i = 0; i < numResidualBlocks; i++) {
    const shortcut = x;
    let h = tf.layers
      .dense({ units: hiddenSize, activation: 'relu', name: `${namePrefix}_res${i}_a` })
      .apply(x);
    h = tf.layers
      .dense({ units: hiddenSize, name: `${namePrefix}_res${i}_b` })
      .apply(h);
    x = tf.layers.add({ name: `${namePrefix}_res${i}_add` }).apply([h, shortcut]);
    x = tf.layers.activation({ activation: 'relu', name: `${namePrefix}_res${i}_relu` }).apply(x);
  }

  const headNames = Object.keys(outputHeads);
  const outputs = headNames.map((name) => {
    const { size, activation } = outputHeads[name];
    return tf.layers
      .dense({ units: size, activation: activation || 'linear', name: `${namePrefix}_${name}` })
      .apply(x);
  });

  const model = tf.model({ inputs: input, outputs, name: namePrefix });
  return { model, headNames };
}

/**
 * Runs a built model and returns its outputs as a named object instead of
 * a bare array, e.g. { policyLogits: tensor, value: tensor }.
 */
function callNamed(built, inputTensor) {
  const result = built.model.apply(inputTensor);
  const outputsArray = Array.isArray(result) ? result : [result];
  const named = {};
  built.headNames.forEach((name, i) => {
    named[name] = outputsArray[i];
  });
  return named;
}

class StochasticMuZeroNetwork {
  constructor(config) {
    this.config = config;
    const { observationSize, numActions, codebookSize, hiddenSize, numResidualBlocks } = config;

    // h: observation -> hidden state
    this.representationNet = buildResidualMLP(
      observationSize,
      { state: { size: hiddenSize } },
      hiddenSize,
      numResidualBlocks,
      'representation'
    );

    // f: hidden state -> policy logits + value
    this.predictionNet = buildResidualMLP(
      hiddenSize,
      { policyLogits: { size: numActions }, value: { size: 1 } },
      hiddenSize,
      numResidualBlocks,
      'prediction'
    );

    // phi: [hidden state, action one-hot] -> afterstate
    this.afterstateDynamicsNet = buildResidualMLP(
      hiddenSize + numActions,
      { afterstate: { size: hiddenSize } },
      hiddenSize,
      numResidualBlocks,
      'afterstate_dynamics'
    );

    // psi: afterstate -> chance-outcome logits + Q-value
    this.afterstatePredictionNet = buildResidualMLP(
      hiddenSize,
      { chanceLogits: { size: codebookSize }, qValue: { size: 1 } },
      hiddenSize,
      numResidualBlocks,
      'afterstate_prediction'
    );

    // g: [afterstate, chance-outcome one-hot] -> next hidden state + reward
    this.dynamicsNet = buildResidualMLP(
      hiddenSize + codebookSize,
      { nextState: { size: hiddenSize }, reward: { size: 1 } },
      hiddenSize,
      numResidualBlocks,
      'dynamics'
    );

    // e: real future observation -> chance-outcome logits (training only —
    // this is how we figure out "what actually happened, in code form" so
    // psi has something to learn to predict).
    this.encoderNet = buildResidualMLP(
      observationSize,
      { chanceLogits: { size: codebookSize } },
      hiddenSize,
      numResidualBlocks,
      'encoder'
    );

    // Building each model with a dummy zero input forces TensorFlow.js to
    // actually allocate all the weight variables right now, rather than
    // lazily on first real use. This makes sure the optimizer sees every
    // trainable variable from the very first training step.
    this._warmUp();
  }

  _warmUp() {
    tf.tidy(() => {
      const { observationSize, numActions, codebookSize, hiddenSize } = this.config;
      const obs = tf.zeros([1, observationSize]);
      const state = callNamed(this.representationNet, obs).state;
      callNamed(this.predictionNet, state);
      const actionOneHot = tf.zeros([1, numActions]);
      const afterstate = callNamed(
        this.afterstateDynamicsNet,
        tf.concat([state, actionOneHot], 1)
      ).afterstate;
      callNamed(this.afterstatePredictionNet, afterstate);
      const chanceOneHot = tf.zeros([1, codebookSize]);
      callNamed(this.dynamicsNet, tf.concat([afterstate, chanceOneHot], 1));
      callNamed(this.encoderNet, obs);
    });
  }

  // ---------------------------------------------------------------
  // Batched functions (tensors in, tensors out). These are the ones
  // used during training, where we process many examples at once.
  // ---------------------------------------------------------------

  /** h: [B, observationSize] -> [B, hiddenSize] */
  representation(observationBatch) {
    return callNamed(this.representationNet, observationBatch).state;
  }

  /** f: [B, hiddenSize] -> { policyLogits: [B, numActions], value: [B, 1] } */
  prediction(stateBatch) {
    return callNamed(this.predictionNet, stateBatch);
  }

  /** phi: [B, hiddenSize], [B, numActions] one-hot -> [B, hiddenSize] */
  afterstateDynamics(stateBatch, actionOneHotBatch) {
    const input = tf.concat([stateBatch, actionOneHotBatch], 1);
    return callNamed(this.afterstateDynamicsNet, input).afterstate;
  }

  /** psi: [B, hiddenSize] -> { chanceLogits: [B, codebookSize], qValue: [B, 1] } */
  afterstatePrediction(afterstateBatch) {
    return callNamed(this.afterstatePredictionNet, afterstateBatch);
  }

  /** g: [B, hiddenSize], [B, codebookSize] one-hot -> { nextState, reward } */
  dynamics(afterstateBatch, chanceOneHotBatch) {
    const input = tf.concat([afterstateBatch, chanceOneHotBatch], 1);
    return callNamed(this.dynamicsNet, input);
  }

  /** e: [B, observationSize] -> { chanceLogits: [B, codebookSize] } (training only) */
  encodeChance(observationBatch) {
    return callNamed(this.encoderNet, observationBatch).chanceLogits;
  }

  // ---------------------------------------------------------------
  // Single-example convenience wrappers (plain JS arrays in, plain JS
  // numbers/arrays out). These are used inside MCTS, where we're always
  // dealing with one imagined position at a time and it's much simpler
  // to work with plain arrays than to juggle tensors through a tree
  // search. Each call is wrapped in tf.tidy so we don't leak memory.
  // ---------------------------------------------------------------

  representationSingle(observationArray) {
    return tf.tidy(() => {
      const obsTensor = tf.tensor2d([observationArray]);
      const state = this.representation(obsTensor);
      return Array.from(state.dataSync());
    });
  }

  predictionSingle(stateArray) {
    return tf.tidy(() => {
      const stateTensor = tf.tensor2d([stateArray]);
      const { policyLogits, value } = this.prediction(stateTensor);
      return {
        policyLogits: Array.from(policyLogits.dataSync()),
        value: value.dataSync()[0],
      };
    });
  }

  afterstateDynamicsSingle(stateArray, actionIndex) {
    return tf.tidy(() => {
      const stateTensor = tf.tensor2d([stateArray]);
      const actionOneHot = tf.oneHot([actionIndex], this.config.numActions);
      const afterstate = this.afterstateDynamics(stateTensor, actionOneHot);
      return Array.from(afterstate.dataSync());
    });
  }

  afterstatePredictionSingle(afterstateArray) {
    return tf.tidy(() => {
      const afterstateTensor = tf.tensor2d([afterstateArray]);
      const { chanceLogits, qValue } = this.afterstatePrediction(afterstateTensor);
      return {
        chanceLogits: Array.from(chanceLogits.dataSync()),
        qValue: qValue.dataSync()[0],
      };
    });
  }

  dynamicsSingle(afterstateArray, chanceIndex) {
    return tf.tidy(() => {
      const afterstateTensor = tf.tensor2d([afterstateArray]);
      const chanceOneHot = tf.oneHot([chanceIndex], this.config.codebookSize);
      const { nextState, reward } = this.dynamics(afterstateTensor, chanceOneHot);
      return {
        nextState: Array.from(nextState.dataSync()),
        reward: reward.dataSync()[0],
      };
    });
  }

  /** Save all five networks' weights to a folder (browser: IndexedDB, node: file://path). */
  async save(pathPrefix) {
    await this.representationNet.model.save(`${pathPrefix}/representation`);
    await this.predictionNet.model.save(`${pathPrefix}/prediction`);
    await this.afterstateDynamicsNet.model.save(`${pathPrefix}/afterstate_dynamics`);
    await this.afterstatePredictionNet.model.save(`${pathPrefix}/afterstate_prediction`);
    await this.dynamicsNet.model.save(`${pathPrefix}/dynamics`);
    await this.encoderNet.model.save(`${pathPrefix}/encoder`);
  }
}

module.exports = { StochasticMuZeroNetwork, buildResidualMLP, callNamed };
