// windyLineEnv.js
//
// A tiny, deliberately simple stochastic environment used to demonstrate
// the library end-to-end. It's small enough to reason about by hand,
// while still having a genuine "afterstate" split between what the agent
// controls and what chance controls — which is exactly what Stochastic
// MuZero is built to handle.
//
// The game: a token sits on a line of `size` cells, numbered 0..size-1,
// starting in the middle. Each turn:
//   1. The agent moves left, stays, or moves right (its CHOICE).
//      This produces the "afterstate" — where the token is right after
//      the agent's move, but before the wind has had a say.
//   2. A random gust of wind then pushes the token left, right, or not at
//      all (CHANCE — the agent has no control over this).
// Reaching the rightmost cell is a win (+1 reward), reaching the leftmost
// cell is a loss (-1 reward), and the episode times out with 0 reward if
// neither happens quickly enough.
//
// Because the wind is random and independent of the agent's move, a
// model that only predicts ONE next state per action (like plain MuZero)
// can't represent this well — it has to predict a distribution over
// several possible next states, which is exactly what the afterstate +
// chance-outcome split gives us.

class WindyLineEnv {
  constructor({ size = 5, maxSteps = 20 } = {}) {
    this.size = size;
    this.maxSteps = maxSteps;
  }

  /** Starts a new episode with the token in the middle, returns the first observation. */
  reset() {
    this.position = Math.floor(this.size / 2);
    this.stepsTaken = 0;
    return this._observation();
  }

  /** All three moves are always legal in this simple game. */
  legalActions() {
    return [0, 1, 2]; // 0 = move left, 1 = stay, 2 = move right
  }

  /** Applies one turn: the agent's move, then a random gust of wind. */
  step(action) {
    const move = action - 1; // turns {0,1,2} into {-1,0,+1}
    const afterAgentMove = this._clamp(this.position + move);

    // The chance event: a 25% chance of a leftward gust, 25% rightward,
    // 50% calm. This is the part of the transition the agent can't
    // control or predict with certainty.
    const roll = Math.random();
    let wind = 0;
    if (roll < 0.25) wind = -1;
    else if (roll < 0.5) wind = 1;

    this.position = this._clamp(afterAgentMove + wind);
    this.stepsTaken += 1;

    let reward = 0;
    let done = false;
    if (this.position === this.size - 1) {
      reward = 1;
      done = true;
    } else if (this.position === 0) {
      reward = -1;
      done = true;
    } else if (this.stepsTaken >= this.maxSteps) {
      done = true;
    }

    return { observation: this._observation(), reward, done };
  }

  _clamp(position) {
    return Math.max(0, Math.min(this.size - 1, position));
  }

  /** One-hot encoding of the token's position — a simple, fixed-length observation vector. */
  _observation() {
    const obs = new Array(this.size).fill(0);
    obs[this.position] = 1;
    return obs;
  }
}

module.exports = { WindyLineEnv };
