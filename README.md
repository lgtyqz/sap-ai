# Stochastic MuZero in TensorFlow.js

A from-scratch, heavily-commented implementation of **Stochastic MuZero**
(Antonoglou, Schrittwieser, Ozair, Hubert & Silver, ICLR 2022), meant to be
readable by someone who understands the algorithm conceptually but hasn't
necessarily built one before. Every file has comments explaining not just
*what* the code does but *why*, in plain English.

You bring the environment; this brings the agent.

## Installation

This code needs TensorFlow.js as a dependency, which isn't bundled here.

```bash
npm install
```

By default this installs the pure-JavaScript `@tensorflow/tfjs`, which
works anywhere (browser or Node) but is slower. If you're running in
Node.js and want much faster training, also install the native bindings:

```bash
npm install @tensorflow/tfjs-node
```

...and add `require('@tensorflow/tfjs-node');` at the very top of your
entry script, before anything else touches TensorFlow. (No other code
changes needed — it patches the same `@tensorflow/tfjs` API underneath.)

## Quick start

```bash
npm run example
```

This plays a toy stochastic game (`example/windyLineEnv.js` — a token on a
line, buffeted by random gusts of wind) against itself, training as it
goes. You should see the average reward climb from around 0 towards +1
over a few hundred episodes.

## How the pieces fit together

| File | Role |
|---|---|
| `src/config.js` | All hyperparameters in one place. **You'll need to override the environment-shape settings** (`observationSize`, `numActions`, `codebookSize`) to match your own game. |
| `src/network.js` | The five neural network functions the algorithm is built from (see below). |
| `src/mcts.js` | Monte Carlo Tree Search with alternating decision/chance nodes — the "planning" step. |
| `src/replayBuffer.js` | Stores finished games and builds training batches (including n-step value targets). |
| `src/losses.js` | The loss functions that grade each prediction against reality. |
| `src/train.js` | Unrolls the model through several real steps and takes a gradient step across all five networks at once. |
| `src/selfplay.js` | Plays a game against your environment using MCTS, recording a trajectory. |
| `src/utils.js` | Generic helpers: sampling, softmax, the "straight-through" and "gradient scaling" tricks. |
| `src/index.js` | The main entry point — `createAgent(config)` wires everything above together. |
| `example/` | A tiny toy environment and a script that trains an agent on it end to end. |

### The five functions, and what they're called in the code

If you've read about (Stochastic) MuZero, this is the vocabulary:

- **h — representation** (`network.representation`): real observation → hidden state.
- **f — prediction** (`network.prediction`): hidden state → policy + value.
- **φ (phi) — afterstate dynamics** (`network.afterstateDynamics`): hidden state + action → *afterstate* (the imagined situation right after you've acted, but before chance has had its say).
- **ψ (psi) — afterstate prediction** (`network.afterstatePrediction`): afterstate → a distribution over what chance might do next, plus a Q-value.
- **g — dynamics** (`network.dynamics`): afterstate + a chance outcome → next hidden state + reward.
- **e — encoder** (`network.encodeChance`, training only): looks at what *actually* happened next in a real game and turns it into a discrete "chance code," which is what ψ is trained to predict in advance.

This is the one addition Stochastic MuZero makes on top of regular
MuZero: splitting each transition into "you act" → afterstate → "chance
resolves" → next state, so the model can represent a whole distribution
of possible outcomes instead of being forced to predict one single
(deterministic) future.

## Plugging in your own environment

Your environment just needs three methods:

```js
class MyEnvironment {
  reset() {
    // Start a new episode. Return the first observation as a plain
    // array/list of numbers, of length config.observationSize.
    return observation;
  }

  step(action) {
    // Apply `action` (an integer, 0 .. config.numActions - 1).
    // Return { observation, reward, done }.
    return { observation, reward, done };
  }

  legalActions() {
    // Which action indices are legal right now. Return all of them if
    // every action is always legal.
    return [0, 1, 2, /* ... */];
  }
}
```

Then:

```js
const { createAgent } = require('./src/index');

const agent = createAgent({
  observationSize: /* length of your observation array */,
  numActions: /* how many actions your game has */,
  codebookSize: /* how many distinct "flavors" of randomness to allow —
                   start small (e.g. 4-8) and increase if the model seems
                   to be running out of room to represent your game's
                   randomness */,
});

for (let episode = 0; episode < 10000; episode++) {
  const trajectory = agent.selfPlayer.playGame(new MyEnvironment());
  agent.replayBuffer.saveGame(trajectory);

  if (agent.replayBuffer.size() >= agent.config.batchSize) {
    const loss = agent.trainer.trainStep(agent.replayBuffer);
  }
}
```

If your game is purely deterministic, this still works fine — the model
will simply learn that the chance-outcome distribution is very "peaked"
on one outcome (the paper confirms this happens naturally in Go, a
deterministic game, when they tested Stochastic MuZero there).

## Simplifications made versus the paper

This implementation is faithful to the core algorithm — the afterstate
split, the five/six functions, the chance loss, gradient scaling, the
alternating decision/chance search — but a few things were simplified for
readability and to avoid needing a huge amount of environment-specific
tuning code:

- **Chance codes use a Gumbel-softmax relaxation with a straight-through
  estimator**, rather than the paper's formal VQ-VAE codebook with a
  commitment loss. This is a common, simpler alternative for learning a
  discrete latent code end-to-end, and plays the same role: a discrete
  "chance code" that's still differentiable enough to train.
- **Value and reward are plain scalars** (trained with mean-squared
  error), rather than the paper's categorical/distributional
  representation (predicting a distribution over many possible value
  "bins"). The categorical approach tends to help with very large or
  widely-varying rewards (like 2048's scores running into the hundreds
  of thousands); if your rewards are modest in scale, scalars work well
  and are far simpler to reason about.
- **Uniform replay sampling**, not the paper's prioritized replay (which
  samples more often from positions where the model was most surprised).
- **Single-player assumption.** The search and backpropagation code
  doesn't flip the sign of values between players. If you want to model
  a two-player, zero-sum game, you'll want `backpropagate` in `mcts.js`
  to negate the value when moving between nodes belonging to different
  players — the paper's original MuZero pseudocode shows this pattern
  if you want a reference.
- **No autoregressive/sampled action heads** for very large or
  combinatorial action spaces (the paper uses this for backgammon's
  action space). Plain discrete actions are assumed; this covers most
  custom environments.

None of these change the core ideas — they're the kind of simplifications
you'd want to peel back one at a time once you have something training
end-to-end on your own environment and want to push performance further.

## A note on testing

This code was written and syntax-checked in an environment without
network access, so it hasn't been run against a live TensorFlow.js
install. Read through `example/run.js` and `example/windyLineEnv.js`
first when you try it — they're the simplest possible end-to-end check
that everything is wired up correctly on your machine.
