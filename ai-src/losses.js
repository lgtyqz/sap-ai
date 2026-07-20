// losses.js
//
// The "grading" functions that tell the networks how wrong their
// predictions were, so gradient descent knows which way to nudge the
// weights. Each one takes a `mask` tensor (1 = real position, 0 = padded
// position past the end of a game) so padded positions don't contribute
// to learning at all.

const tf = require('@tensorflow/tfjs');
const { stopGradient } = require('./utils');

/**
 * Cross-entropy between a predicted policy (given as logits) and a target
 * policy (given as probabilities, e.g. MCTS visit counts). This is the
 * standard way to train a network to match a target probability
 * distribution: it penalizes the model more the more "surprised" it
 * would be to see the target distribution, given what it currently
 * predicts.
 */
function policyLoss(predictedLogits, targetProbabilities, mask) {
  return tf.tidy(() => {
    const logProbs = tf.logSoftmax(predictedLogits);
    // Cross-entropy for each example: -sum(target * log(predicted)).
    const perExample = tf.neg(tf.sum(tf.mul(targetProbabilities, logProbs), -1));
    return maskedMean(perExample, mask);
  });
}

/** Mean-squared-error, used for value, Q-value, and reward predictions (all plain scalars here). */
function scalarLoss(predicted, target, mask) {
  return tf.tidy(() => {
    const perExample = tf.square(tf.sub(tf.squeeze(predicted, [-1]), target));
    return maskedMean(perExample, mask);
  });
}

/**
 * Cross-entropy between psi's predicted chance-outcome distribution and
 * the (stop-gradient) code actually produced by the encoder from the real
 * future observation. This is what trains psi to "guess what chance will
 * do" without being allowed to peek at the future itself.
 */
function chanceLoss(predictedLogits, targetCode, mask) {
  return tf.tidy(() => {
    const logProbs = tf.logSoftmax(predictedLogits);
    const perExample = tf.neg(tf.sum(tf.mul(stopGradient(targetCode), logProbs), -1));
    return maskedMean(perExample, mask);
  });
}

/** Averages a per-example loss, ignoring (masking out) padded examples. */
function maskedMean(perExampleLoss, mask) {
  const masked = tf.mul(perExampleLoss, mask);
  const total = tf.sum(masked);
  const count = tf.add(tf.sum(mask), 1e-8); // avoid divide-by-zero if a batch is all padding
  return tf.div(total, count);
}

module.exports = { policyLoss, scalarLoss, chanceLoss };
