// twentyFortyEightEnv.js
//
// A recreation of the 2048 environment used in the Stochastic MuZero
// paper (Antonoglou et al., 2022, Appendix C), wired to the same
// { reset(), step(action), legalActions() } interface the rest of this
// library expects.
//
// A quick reminder of the rules, in case it's been a while: you have a
// 4x4 board of numbered tiles. Each turn you slide every tile as far as
// possible in one direction (up/down/left/right); tiles of equal value
// that collide merge into one tile worth their sum, and each merge adds
// that new tile's value to your score. After you slide, the game drops a
// new tile (a 2, 90% of the time, or a 4, 10% of the time) onto a random
// empty square. The game ends when the board is full and no more merges
// are possible.
//
// If you've read through mcts.js: this game is a really clean real-world
// example of an "afterstate". Your slide-and-merge is entirely your own
// deterministic choice — that's the afterstate. The random tile that
// appears afterward is chance's turn, completely outside your control.
// That's exactly the two-step split Stochastic MuZero's model is built
// around.
//
// Per the paper's Appendix C, this uses:
//   - a 4x4 board (16 cells)
//   - a 31-bit binary encoding of each cell's value (16 * 31 = 496 total
//     numbers in the observation — matches config.observationSize: 496)
//   - a codebook size of 32 for the chance outcomes (matches
//     config.codebookSize: 32 — one code per (empty-cell-slot, tile-value)
//     combination, of which there are up to 16 cells * 2 tile values)
//   - reward = the value of every tile created by a merge, summed over
//     the whole episode (so cumulative reward across an episode equals
//     the game's final score)

const BOARD_SIZE = 4;
const NUM_CELLS = BOARD_SIZE * BOARD_SIZE;
const BITS_PER_CELL = 31;

// Action indices. Pick whichever order you like — this is just the
// convention used inside this file and in legalActions()'s output.
const ACTIONS = { UP: 0, RIGHT: 1, DOWN: 2, LEFT: 3 };

class TwentyFortyEightEnv {
  /**
   * @param {Object} options
   * @param {'binary'|'flat'} options.encoding - 'binary' reproduces the
   *   paper's 496-number observation exactly. 'flat' is a much smaller
   *   16-number observation (just each cell's raw value) that's handy
   *   for quick experiments or debugging, at the cost of not matching
   *   the paper.
   */
  constructor({ encoding = 'binary' } = {}) {
    this.encoding = encoding;
    this.board = null;
    this.score = 0;
  }

  /** The observation length you should set config.observationSize to, for the chosen encoding. */
  static observationSize(encoding = 'binary') {
    return encoding === 'binary' ? NUM_CELLS * BITS_PER_CELL : NUM_CELLS;
  }

  reset() {
    this.board = new Array(NUM_CELLS).fill(0);
    this.score = 0;
    spawnRandomTile(this.board);
    spawnRandomTile(this.board);
    return this._observation();
  }

  /** Returns which of the 4 directions would actually change the board right now. */
  legalActions() {
    const legal = [];
    for (const action of [ACTIONS.UP, ACTIONS.RIGHT, ACTIONS.DOWN, ACTIONS.LEFT]) {
      const { moved } = applyMove(this.board, action);
      if (moved) legal.push(action);
    }
    return legal;
  }

  step(action) {
    const { newBoard, reward, moved } = applyMove(this.board, action);

    if (!moved) {
      // Shouldn't happen if you only ever pass actions from
      // legalActions(), but handled gracefully just in case: nothing
      // changes, no tile spawns, no reward, and this doesn't end the game.
      return { observation: this._observation(), reward: 0, done: false };
    }

    this.board = newBoard;
    this.score += reward;
    spawnRandomTile(this.board); // the "chance" half of the transition

    const done = this.legalActions().length === 0;
    return { observation: this._observation(), reward, done };
  }

  /** A human-readable printout of the current board, handy for debugging. */
  render() {
    let out = '';
    for (let r = 0; r < BOARD_SIZE; r++) {
      const row = getRow(this.board, r).map((v) => (v === 0 ? '.' : String(v)).padStart(6));
      out += row.join(' ') + '\n';
    }
    return out;
  }

  _observation() {
    return this.encoding === 'binary' ? encodeBinary(this.board) : this.board.slice();
  }

  /**
   * The paper scales value/reward targets through this invertible
   * transform before training, since 2048's scores can run into the
   * hundreds of thousands and raw values that large are hard for a
   * network to regress accurately. Apply this to rewards yourself before
   * feeding them to the training code if you run into that problem; it's
   * not applied automatically since this library's default losses are
   * plain scalar MSE (see the "simplifications" note in the main README).
   */
  static scaleValue(x, epsilon = 0.001) {
    const sign = Math.sign(x);
    return sign * (Math.sqrt(Math.abs(x) + 1) - 1 + epsilon * Math.abs(x));
  }

  /** Inverse of scaleValue, for turning a scaled prediction back into a real score. */
  static unscaleValue(x, epsilon = 0.001) {
    const sign = Math.sign(x);
    const target = Math.abs(x);
    if (target === 0) return 0;
    // Inverting sign*(sqrt(|x|+1) - 1 + eps*|x|) has no closed form for
    // general epsilon, so we solve it numerically with bisection instead
    // (plenty accurate here, and this only ever runs on a handful of
    // numbers at a time). Since forward(x) >= eps*x - 1, we know
    // x <= (target + 1) / eps, which gives us a safe upper bound to
    // start the search from.
    let lo = 0;
    let hi = (target + 2) / epsilon;
    for (let i = 0; i < 200; i++) {
      const mid = (lo + hi) / 2;
      const forward = Math.sqrt(mid + 1) - 1 + epsilon * mid;
      if (forward < target) lo = mid;
      else hi = mid;
    }
    return sign * ((lo + hi) / 2);
  }
}

// ----------------------------------------------------------------------
// Board mechanics — pure functions, no side effects, so they're easy to
// test and reuse (e.g. legalActions() calls applyMove() without mutating
// the real board).
// ----------------------------------------------------------------------

/** Slides+merges `board` one step in `action`'s direction. Never mutates `board`. */
function applyMove(board, action) {
  const newBoard = board.slice();
  let reward = 0;

  if (action === ACTIONS.LEFT || action === ACTIONS.RIGHT) {
    for (let r = 0; r < BOARD_SIZE; r++) {
      let line = getRow(newBoard, r);
      if (action === ACTIONS.RIGHT) line = line.reverse();
      const result = slideAndMergeLine(line);
      reward += result.reward;
      const finalLine = action === ACTIONS.RIGHT ? result.line.reverse() : result.line;
      setRow(newBoard, r, finalLine);
    }
  } else {
    for (let c = 0; c < BOARD_SIZE; c++) {
      let line = getCol(newBoard, c);
      if (action === ACTIONS.DOWN) line = line.reverse();
      const result = slideAndMergeLine(line);
      reward += result.reward;
      const finalLine = action === ACTIONS.DOWN ? result.line.reverse() : result.line;
      setCol(newBoard, c, finalLine);
    }
  }

  const moved = !boardsEqual(board, newBoard);
  return { newBoard, reward, moved };
}

/**
 * Slides one row/column (already oriented so "towards index 0" is the
 * direction of travel) as far as it can go, merging equal adjacent tiles.
 * Each tile can only take part in one merge per move, which is why we
 * walk through the compacted tiles once, left to right.
 */
function slideAndMergeLine(line) {
  const nonZero = line.filter((v) => v !== 0);
  const merged = [];
  let reward = 0;
  let i = 0;
  while (i < nonZero.length) {
    if (i + 1 < nonZero.length && nonZero[i] === nonZero[i + 1]) {
      const mergedValue = nonZero[i] * 2;
      merged.push(mergedValue);
      reward += mergedValue;
      i += 2;
    } else {
      merged.push(nonZero[i]);
      i += 1;
    }
  }
  while (merged.length < BOARD_SIZE) merged.push(0);
  return { line: merged, reward };
}

/** Drops a new tile (2 with 90% probability, 4 with 10%) onto a random empty cell. This is the "chance" step. */
function spawnRandomTile(board) {
  const emptyIndices = [];
  for (let i = 0; i < board.length; i++) {
    if (board[i] === 0) emptyIndices.push(i);
  }
  if (emptyIndices.length === 0) return;
  const index = emptyIndices[Math.floor(Math.random() * emptyIndices.length)];
  board[index] = Math.random() < 0.9 ? 2 : 4;
}

function getRow(board, r) {
  return [board[r * 4], board[r * 4 + 1], board[r * 4 + 2], board[r * 4 + 3]];
}

function setRow(board, r, values) {
  for (let c = 0; c < BOARD_SIZE; c++) board[r * 4 + c] = values[c];
}

function getCol(board, c) {
  return [board[c], board[4 + c], board[8 + c], board[12 + c]];
}

function setCol(board, c, values) {
  for (let r = 0; r < BOARD_SIZE; r++) board[r * 4 + c] = values[r];
}

function boardsEqual(a, b) {
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** Turns each cell's value into a 31-bit binary vector, matching the paper's observation encoding. */
function encodeBinary(board) {
  const obs = new Array(NUM_CELLS * BITS_PER_CELL);
  let offset = 0;
  for (let i = 0; i < board.length; i++) {
    const value = board[i];
    for (let bit = 0; bit < BITS_PER_CELL; bit++) {
      obs[offset++] = (value >> bit) & 1;
    }
  }
  return obs;
}

module.exports = { TwentyFortyEightEnv, ACTIONS };

// ----------------------------------------------------------------------
// Quick self-test: run `node twentyFortyEightEnv.js` directly (no
// TensorFlow / no other project files needed) to watch a random game play
// out and sanity-check the rules above.
// ----------------------------------------------------------------------
if (require.main === module) {
  const env = new TwentyFortyEightEnv({ encoding: 'flat' });
  env.reset();
  console.log('Starting board:');
  console.log(env.render());

  let done = false;
  let turns = 0;
  while (!done && turns < 10000) {
    const legal = env.legalActions();
    const action = legal[Math.floor(Math.random() * legal.length)];
    const result = env.step(action);
    done = result.done;
    turns += 1;
  }

  console.log(`Game over after ${turns} turns. Final score: ${env.score}`);
  console.log(env.render());
  console.log(
    `observationSize('binary') = ${TwentyFortyEightEnv.observationSize('binary')}, ` +
      `observationSize('flat') = ${TwentyFortyEightEnv.observationSize('flat')}`
  );
}
