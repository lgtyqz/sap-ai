// config.js
//
// This file is the "settings panel" for the whole algorithm. Every number
// that controls how big the networks are, how far ahead the search looks,
// or how fast it learns lives here so you don't have to go hunting through
// the code to tune things.
//
// You WILL need to change the first block (environment shape) to match
// whatever game/environment you build. Everything else has reasonable
// defaults to start experimenting with.

module.exports = {
  // ------------------------------------------------------------------
  // Environment shape — these describe the game you're plugging in.
  // ------------------------------------------------------------------

  // How many numbers describe a single observation (game state) from your
  // environment. E.g. if your game state is a flattened 4x4 board, that's 16.
  observationSize: 8,

  // How many distinct actions the agent can choose from at any point.
  numActions: 4,

  // How many distinct "flavors" of randomness the model is allowed to
  // imagine. Think of this as "how many different dice-roll outcomes can
  // the model tell apart". Bigger environments with richer randomness
  // (e.g. backgammon's 21 possible dice rolls) need a bigger codebook;
  // simple coin-flip style randomness might only need 2-4.
  codebookSize: 8,

  // ------------------------------------------------------------------
  // Network size — how big the neural networks are.
  // ------------------------------------------------------------------

  // The length of the internal "hidden state" vector that all five
  // functions pass around. Bigger = more capacity to represent complex
  // situations, but slower to train.
  hiddenSize: 64,

  // How many residual blocks (small stacks of layers with a shortcut
  // connection) each network has. More blocks = more capacity.
  numResidualBlocks: 2,

  // ------------------------------------------------------------------
  // Self-play / search — how MCTS explores imagined futures.
  // ------------------------------------------------------------------

  // How many imagined steps into the future MCTS simulates before the
  // agent commits to a real move. More simulations = stronger play, but
  // slower self-play.
  numSimulations: 30,

  // How much a reward one step further into the future is "worth" compared
  // to an immediate reward. Closer to 1 = very patient/long-term thinking.
  discount: 0.997,

  // Exploration noise added to the root of the search tree so self-play
  // doesn't always play the exact same moves (which would prevent learning
  // anything new). Alpha controls the "shape" of the noise, fraction
  // controls how much of it gets mixed in.
  rootDirichletAlpha: 0.25,
  rootExplorationFraction: 0.25,

  // Constants used inside the UCB formula that balances trying new actions
  // ("exploration") against playing known-good ones ("exploitation"). The
  // defaults are the same ones used in the original MuZero/AlphaZero papers.
  pbCBase: 19652,
  pbCInit: 1.25,

  // Temperature for turning MCTS visit counts into a move. 1.0 = sample
  // proportionally to how often each action was visited (more exploratory),
  // values near 0 = always pick the most-visited action (more greedy).
  // You can pass a custom schedule into the Actor if you want this to
  // decay over the course of training.
  actionSelectionTemperature: 1.0,

  // Temperature for the Gumbel-softmax trick used to draw a differentiable
  // "chance code" sample during training (see network.js). Lower = closer
  // to a true hard one-hot pick, higher = softer/blurrier.
  gumbelTemperature: 1.0,

  // ------------------------------------------------------------------
  // Replay buffer — where finished games are stored for training.
  // ------------------------------------------------------------------

  // Maximum number of games kept in memory. Oldest games are dropped first.
  bufferSize: 2000,

  // How many training positions are grouped together into one gradient step.
  batchSize: 32,

  // How many real steps into the future the model is "unrolled" during
  // training (this is the K in "predict k steps ahead and check against
  // what really happened").
  numUnrollSteps: 5,

  // How many real steps ahead we look before bootstrapping off the
  // network's own value prediction, when building a value training target
  // (an "n-step return", a standard reinforcement-learning technique).
  tdSteps: 10,

  // ------------------------------------------------------------------
  // Optimization
  // ------------------------------------------------------------------

  learningRate: 1e-3,

  // How much each unrolled step's gradient is scaled down before flowing
  // further back into earlier parts of the model. This keeps a long
  // unroll from overwhelming the shared representation/afterstate/dynamics
  // networks with an oversized combined gradient signal.
  gradientScaleForUnroll: 0.5,

  // ------------------------------------------------------------------
  // How much each piece of the loss counts towards the total.
  // ------------------------------------------------------------------
  valueLossWeight: 0.25,
  policyLossWeight: 1.0,
  rewardLossWeight: 1.0,
  chanceLossWeight: 1.0,
  afterstateValueLossWeight: 0.25,
};
