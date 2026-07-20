// utils.js
//
// Small, generic helper functions used all over the codebase: turning
// numbers into probabilities, sampling from probability distributions,
// and a couple of "gradient trick" helpers that let us do things like
// "use a discrete choice in the forward pass, but let gradients flow as
// if it were continuous" (this sounds exotic but it's a well-known trick
// called the "straight-through estimator").
//
// None of this is specific to Stochastic MuZero — it's just plumbing.

const tf = require('@tensorflow/tfjs');

/**
 * Turn a plain JS array of numbers ("logits") into a probability
 * distribution that sums to 1, the standard "softmax" operation.
 * Works on the CPU with plain arrays (used inside MCTS, which deals with
 * raw numbers rather than tensors for simplicity).
 */
function softmaxArray(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

/** Index of the largest value in a plain JS array. */
function argmaxArray(values) {
  let bestIndex = 0;
  let bestValue = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > bestValue) {
      bestValue = values[i];
      bestIndex = i;
    }
  }
  return bestIndex;
}

/**
 * Randomly pick one of `keys[i]` with probability `probs[i]`.
 * This is how MCTS "rolls the dice" at chance nodes, and how self-play
 * picks a real move from the search's visit-count distribution.
 */
function sampleCategorical(keys, probs) {
  const r = Math.random();
  let cumulative = 0;
  for (let i = 0; i < keys.length; i++) {
    cumulative += probs[i];
    if (r < cumulative) return keys[i];
  }
  // Floating point safety net: if rounding error left us just short of 1,
  // fall back to the last option instead of returning undefined.
  return keys[keys.length - 1];
}

/**
 * Draw a sample from a symmetric Dirichlet distribution, used to add
 * exploration noise to the search's root node. If you haven't met the
 * Dirichlet distribution before: it's a way of randomly generating a set
 * of numbers that add up to 1 (like a randomized probability distribution
 * over your actions), which is exactly what we want to "nudge" the root
 * priors with.
 */
function sampleDirichlet(alpha, size) {
  // A Dirichlet sample can be built by drawing `size` independent Gamma(alpha)
  // samples and normalizing them so they sum to 1.
  const samples = Array.from({ length: size }, () => sampleGamma(alpha));
  const sum = samples.reduce((a, b) => a + b, 0);
  return samples.map((s) => s / sum);
}

/** Marsaglia-Tsang method for sampling from a Gamma(shape, 1) distribution. */
function sampleGamma(shape) {
  if (shape < 1) {
    // Boost small shapes by 1 and correct afterwards (standard trick).
    const u = Math.random();
    return sampleGamma(shape + 1) * Math.pow(u, 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  while (true) {
    let x, v;
    do {
      x = sampleGaussian();
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

/** Standard normal sample via the Box-Muller transform. */
function sampleGaussian() {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Tracks the smallest and largest value estimates seen so far during a
 * search, so we can normalize values into a comparable [0, 1] range inside
 * the UCB formula. Without this, a game with huge scores (e.g. 2048) and a
 * game with tiny scores (e.g. win/loss = +/-1) would need totally
 * different exploration constants.
 */
class MinMaxStats {
  constructor() {
    this.maximum = -Infinity;
    this.minimum = Infinity;
  }

  update(value) {
    this.maximum = Math.max(this.maximum, value);
    this.minimum = Math.min(this.minimum, value);
  }

  normalize(value) {
    if (this.maximum > this.minimum) {
      return (value - this.minimum) / (this.maximum - this.minimum);
    }
    return value;
  }
}

/**
 * Stops gradient computation flowing backward through a tensor without
 * changing its forward value. Used in gradient tricks like straight-through
 * estimators to prevent gradients from flowing through certain components.
 *
 * In the forward pass: returns the tensor unchanged.
 * In the backward pass: no gradients flow through (returns zeros).
 */
function stopGradient(tensor) {
  return tf.tidy(() => {
    // customGrad lets us define exactly what happens in forward and backward passes
    const customStopGradient = tf.customGrad((x) => {
      // Forward pass: return tensor as-is
      // Backward pass (gradFunc): return zeros so no gradients flow through
      return {
        value: x,
        gradFunc: (dy) => tf.zerosLike(x)
      };
    });
    return customStopGradient(tensor);
  });
}

/**
 * The "straight-through" gradient trick: on the way forward, this behaves
 * exactly like `hardTensor`. On the way backward (during backpropagation),
 * gradients flow through as if this had just been `softTensor` instead.
 *
 * Why we need this: we want the dynamics function to see a clean, discrete
 * "this is chance outcome #3" signal (the hard version), but we still need
 * gradients to be able to flow back into the network that produced it (the
 * soft, continuous version) so it can learn. This trick gives us both.
 */
function straightThroughEstimator(hardTensor, softTensor) {
  return tf.tidy(() => {
    const difference = tf.sub(hardTensor, softTensor);
    // stopGradient freezes `difference` during backprop so gradients skip
    // straight past it, as if this whole expression were just `softTensor`.
    return tf.add(stopGradient(difference), softTensor);
  });
}

/**
 * Scales gradients flowing backward through a tensor without changing its
 * forward value at all. Used to stop long unrolls of the model from
 * "overloading" earlier parts of the network with a huge combined
 * gradient signal (the same trick used in the original MuZero paper).
 */
function scaleGradient(tensor, scale) {
  return tf.tidy(() => {
    const stopped = stopGradient(tensor);
    // forward value: stopped*(1-scale) + tensor*scale === tensor (since
    // stopped === tensor in value, just detached for gradient purposes)
    return tf.add(stopped.mul(1 - scale), tensor.mul(scale));
  });
}

/**
 * Draws a differentiable "soft" one-hot sample from a categorical
 * distribution described by `logits`, using the Gumbel-softmax trick.
 * This lets us simulate "randomly picking one of several discrete
 * options" in a way that still allows gradients to flow, which a plain
 * random choice would not.
 */
function gumbelSoftmaxSample(logits, temperature = 1.0) {
  return tf.tidy(() => {
    const uniformNoise = tf.randomUniform(logits.shape, 1e-9, 1.0);
    const gumbelNoise = tf.neg(tf.log(tf.neg(tf.log(uniformNoise))));
    const noisyLogits = tf.add(logits, gumbelNoise);
    return tf.softmax(noisyLogits.div(temperature));
  });
}

/** Turns a batch of probability vectors into hard one-hot vectors (argmax). */
function oneHotArgmax(probs) {
  return tf.tidy(() => {
    const lastAxis = probs.shape.length - 1;
    const indices = probs.argMax(lastAxis);
    return tf.oneHot(indices, probs.shape[lastAxis]);
  });
}

module.exports = {
  softmaxArray,
  argmaxArray,
  sampleCategorical,
  sampleDirichlet,
  MinMaxStats,
  stopGradient,
  straightThroughEstimator,
  scaleGradient,
  gumbelSoftmaxSample,
  oneHotArgmax,
};
