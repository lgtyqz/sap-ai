// replayBuffer.js
//
// Stores finished games (trajectories) so the training loop can sample
// from real past experience, and turns those samples into the exact
// shape the training step needs: a starting observation, the real actions
// that were taken afterward, and "target" values for everything the
// networks are trying to predict (policy, value, reward).
//
// Each trajectory is an array of steps, one per real turn taken in the
// game, each shaped like:
//   {
//     observation:   number[]  the observation BEFORE acting this turn
//     action:        number    the action actually taken
//     reward:        number    the reward received arriving at this turn
//                               (0 for the very first step of a game)
//     searchPolicy:  number[]  MCTS's visit-count policy at this turn
//     searchValue:   number    MCTS's root value estimate at this turn
//   }
// This is exactly what selfplay.js produces.

class ReplayBuffer {
  constructor(config) {
    this.config = config;
    this.games = [];
  }

  /** Adds a finished game's trajectory to the buffer, dropping the oldest game if full. */
  saveGame(trajectory) {
    if (this.games.length >= this.config.bufferSize) {
      this.games.shift();
    }
    this.games.push(trajectory);
  }

  size() {
    return this.games.length;
  }

  /**
   * Computes an n-step return: add up `tdSteps` worth of real, discounted
   * future rewards, then "bootstrap" the rest with the network's own value
   * estimate from back when this position was played. This is the
   * standard reinforcement-learning trick of mixing real observed rewards
   * with a learned estimate, which is usually more accurate than trusting
   * either one alone.
   */
  computeValueTarget(trajectory, index) {
    const { tdSteps, discount } = this.config;
    let value = 0;
    let discountFactor = 1;
    const bootstrapIndex = index + tdSteps;

    for (let i = index + 1; i <= Math.min(bootstrapIndex, trajectory.length - 1); i++) {
      value += discountFactor * trajectory[i].reward;
      discountFactor *= discount;
    }

    if (bootstrapIndex < trajectory.length) {
      value += discountFactor * trajectory[bootstrapIndex].searchValue;
    }

    return value;
  }

  /**
   * Builds one training example starting at position `index` inside
   * `trajectory`, unrolled `numUnrollSteps` steps into the future. If the
   * game ends before we run out of unroll steps, we pad by repeating the
   * final position and mark those padded steps in `mask` so the loss
   * function can ignore them.
   */
  buildSample(trajectory, index) {
    const { numUnrollSteps, numActions, observationSize } = this.config;

    const observation = trajectory[index].observation;
    const actions = [];
    const targetPolicies = [this._policyAt(trajectory, index)];
    const targetValues = [this.computeValueTarget(trajectory, index)];
    const targetRewards = [];
    const futureObservations = [];
    const mask = [1]; // the starting position (k=0) is always real

    for (let k = 1; k <= numUnrollSteps; k++) {
      const stepIndex = index + k;
      const isReal = stepIndex < trajectory.length;

      // The action that was actually taken to get from step k-1 to step k.
      const actionSourceIndex = Math.min(index + k - 1, trajectory.length - 1);
      actions.push(trajectory[actionSourceIndex].action);

      if (isReal) {
        targetPolicies.push(this._policyAt(trajectory, stepIndex));
        targetValues.push(this.computeValueTarget(trajectory, stepIndex));
        targetRewards.push(trajectory[stepIndex].reward);
        futureObservations.push(trajectory[stepIndex].observation);
        mask.push(1);
      } else {
        // Past the end of the game: pad with harmless placeholder values.
        // These get multiplied by 0 via `mask` during loss computation, so
        // their exact values don't matter as long as the shapes line up.
        targetPolicies.push(new Array(numActions).fill(1 / numActions));
        targetValues.push(0);
        targetRewards.push(0);
        futureObservations.push(new Array(observationSize).fill(0));
        mask.push(0);
      }
    }

    return { observation, actions, targetPolicies, targetValues, targetRewards, futureObservations, mask };
  }

  _policyAt(trajectory, index) {
    return trajectory[index].searchPolicy;
  }

  /** Samples `batchSize` random (game, position) pairs and builds a training batch from them. */
  sampleBatch() {
    const { batchSize } = this.config;
    const samples = [];
    for (let i = 0; i < batchSize; i++) {
      const game = this.games[Math.floor(Math.random() * this.games.length)];
      const index = Math.floor(Math.random() * game.length);
      samples.push(this.buildSample(game, index));
    }
    return samples;
  }
}

module.exports = { ReplayBuffer };
