// train.js
//
// This is where the networks actually learn. The core idea, in plain
// English:
//
//   1. Grab a batch of real moments from games the agent has already
//      played (from the replay buffer).
//   2. For each one, feed the real starting observation through the
//      representation function to get a hidden state.
//   3. "Unroll" the model forward using the REAL actions that were
//      actually taken at the time — at each step, predict a policy and
//      value, predict what chance will do, and predict a reward — all
//      using only the model's own imagination (the afterstate/dynamics
//      functions), never touching the real environment again.
//   4. Compare every one of those predictions to what actually happened,
//      and adjust every network's weights (via gradient descent) to
//      reduce the gap.
//
// Because step 3 chains representation -> prediction -> afterstate
// dynamics -> afterstate prediction -> dynamics -> prediction -> ...
// together into one big differentiable computation, a single backward
// pass trains all five networks at once, purely by asking "how do we
// adjust every knob in this chain to make the final predictions more
// accurate?".

const tf = require('@tensorflow/tfjs-node-gpu');
const { policyLoss, scalarLoss, chanceLoss } = require('./losses');
const { gumbelSoftmaxSample, straightThroughEstimator, oneHotArgmax, scaleGradient } = require('./utils');

/** Converts an array of per-example sample objects (from replayBuffer.js) into batched tensors. */
function batchSamples(samples, config) {
  const { numActions, codebookSize, numUnrollSteps } = config;

  const observation = tf.tensor2d(samples.map((s) => s.observation));

  // actions[:, k] = the k-th real action taken after the starting position
  const actions = samples.map((s) => s.actions);

  // targetPolicies[k] is a [B, numActions] tensor for unroll step k (k = 0..numUnrollSteps)
  const targetPolicies = [];
  for (let k = 0; k <= numUnrollSteps; k++) {
    targetPolicies.push(tf.tensor2d(samples.map((s) => s.targetPolicies[k])));
  }

  // targetValues[k] is a [B] tensor for unroll step k
  const targetValues = [];
  for (let k = 0; k <= numUnrollSteps; k++) {
    targetValues.push(tf.tensor1d(samples.map((s) => s.targetValues[k])));
  }

  // targetRewards[k] is a [B] tensor for real step t+k+1 (k = 0..numUnrollSteps-1)
  const targetRewards = [];
  for (let k = 0; k < numUnrollSteps; k++) {
    targetRewards.push(tf.tensor1d(samples.map((s) => s.targetRewards[k])));
  }

  // futureObservations[k] is a [B, observationSize] tensor, the real
  // observation that occurred at t+k+1 (used only to compute chance-code
  // targets for training psi and the encoder).
  const futureObservations = [];
  for (let k = 0; k < numUnrollSteps; k++) {
    futureObservations.push(tf.tensor2d(samples.map((s) => s.futureObservations[k])));
  }

  // mask[k] is a [B] tensor, 1 for real positions and 0 for padding past
  // the end of a game, for k = 0..numUnrollSteps
  const mask = [];
  for (let k = 0; k <= numUnrollSteps; k++) {
    mask.push(tf.tensor1d(samples.map((s) => s.mask[k])));
  }

  return { observation, actions, targetPolicies, targetValues, targetRewards, futureObservations, mask, numActions, codebookSize };
}

/**
 * Runs the full forward pass described at the top of this file and
 * returns a single scalar loss tensor that autograd can differentiate
 * with respect to every network's weights at once.
 */
function computeLoss(network, batch, config) {
  const { numUnrollSteps, gradientScaleForUnroll, gumbelTemperature } = config;
  const {
    valueLossWeight,
    policyLossWeight,
    rewardLossWeight,
    chanceLossWeight,
    afterstateValueLossWeight,
  } = config;

  // --- Step 0: the real starting observation -> initial hidden state ---
  let state = network.representation(batch.observation);
  let { policyLogits, value } = network.prediction(state);

  let totalLoss = tf.mul(policyLoss(policyLogits, batch.targetPolicies[0], batch.mask[0]), policyLossWeight);
  totalLoss = tf.add(
    totalLoss,
    tf.mul(scalarLoss(value, batch.targetValues[0], batch.mask[0]), valueLossWeight)
  );

  // --- Steps 1..K: unroll using the REAL actions that were actually taken ---
  for (let k = 0; k < numUnrollSteps; k++) {
    const actionIndices = tf.tensor1d(
      batch.actions.map((row) => row[k]),
      'int32'
    );
    const actionOneHot = tf.oneHot(actionIndices, batch.numActions);

    // state + action -> afterstate (the "before chance resolves" situation)
    const afterstate = network.afterstateDynamics(state, actionOneHot);
    const { chanceLogits: psiChanceLogits, qValue } = network.afterstatePrediction(afterstate);

    // Figure out what chance actually did, by looking at the REAL future
    // observation. We use the Gumbel-softmax + straight-through trick so
    // this is a genuinely discrete "pick one outcome" choice, but one
    // that still lets gradients flow back into the encoder.
    const encoderLogits = network.encodeChance(batch.futureObservations[k]);
    const softChanceCode = gumbelSoftmaxSample(encoderLogits, gumbelTemperature);
    const hardChanceCode = oneHotArgmax(softChanceCode);
    const chanceCode = straightThroughEstimator(hardChanceCode, softChanceCode);

    // psi is trained to predict that same code WITHOUT having seen the
    // future observation — i.e. to anticipate what chance is likely to do.
    totalLoss = tf.add(
      totalLoss,
      tf.mul(chanceLoss(psiChanceLogits, chanceCode, batch.mask[k + 1]), chanceLossWeight)
    );
    // The afterstate's Q-value is trained towards the same value target
    // used for the state at this point in the unroll.
    totalLoss = tf.add(
      totalLoss,
      tf.mul(
        scalarLoss(qValue, batch.targetValues[k], batch.mask[k]),
        afterstateValueLossWeight
      )
    );

    // afterstate + (the real) chance outcome -> next hidden state + reward
    const { nextState, reward } = network.dynamics(afterstate, chanceCode);
    totalLoss = tf.add(
      totalLoss,
      tf.mul(scalarLoss(reward, batch.targetRewards[k], batch.mask[k + 1]), rewardLossWeight)
    );

    // Scale down the gradient before it flows further back through the
    // chain, so a long unroll doesn't overwhelm the shared networks with
    // an oversized combined gradient signal (same trick as in MuZero).
    state = scaleGradient(nextState, gradientScaleForUnroll);

    ({ policyLogits, value } = network.prediction(state));
    totalLoss = tf.add(
      totalLoss,
      tf.mul(policyLoss(policyLogits, batch.targetPolicies[k + 1], batch.mask[k + 1]), policyLossWeight)
    );
    totalLoss = tf.add(
      totalLoss,
      tf.mul(scalarLoss(value, batch.targetValues[k + 1], batch.mask[k + 1]), valueLossWeight)
    );
  }

  return totalLoss;
}

class Trainer {
  constructor(network, config) {
    this.network = network;
    this.config = config;
    this.optimizer = tf.train.adam(config.learningRate);
  }

  /**
   * Samples one batch from the replay buffer and takes a single gradient
   * step. Returns the loss value (a plain JS number) so you can log or
   * plot training progress.
   */
  trainStep(replayBuffer) {
    const samples = replayBuffer.sampleBatch();
    let lossValue;

    // tf.tidy cleans up all the intermediate tensors created while
    // computing the loss and its gradients, so memory doesn't leak over
    // thousands of training steps.
    tf.tidy(() => {
      const batch = batchSamples(samples, this.config);
      // optimizer.minimize both takes the gradient step AND (when
      // returnCost is true) hands back the loss value it computed, so we
      // can report it without doing a second, wasted forward pass.
      const cost = this.optimizer.minimize(
        () => computeLoss(this.network, batch, this.config),
        /* returnCost */ true
      );
      lossValue = cost.dataSync()[0];
    });

    return lossValue;
  }
}

module.exports = { Trainer, computeLoss, batchSamples };
