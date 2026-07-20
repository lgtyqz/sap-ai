// mcts.js
//
// Monte Carlo Tree Search (MCTS), the "planning" part of Stochastic MuZero.
//
// Regular MCTS (like in AlphaZero) only has one kind of node: "it's your
// turn, pick an action". Stochastic MuZero's search alternates between
// two kinds of nodes:
//
//   DECISION nodes - "it's your turn, pick an action" (just like normal
//                     MCTS). Stores a hidden STATE. Expanded using the
//                     dynamics function `g` (or, at the root, using the
//                     representation function `h`).
//
//   CHANCE nodes    - "randomness happens now" (a dice roll, a random
//                     tile spawning, etc). Stores an AFTERSTATE. Expanded
//                     using the afterstate-dynamics function `phi`.
//                     Instead of "picking the best option" like a
//                     decision node would, we SAMPLE an outcome according
//                     to how likely the model thinks each one is —
//                     because the agent doesn't get to choose what chance
//                     does.
//
// A full turn in the tree therefore looks like:
//   decision node --(pick action)--> chance node --(sample outcome)--> decision node --> ...
//
// Every simulation walks down through existing nodes, adds exactly one
// brand new node to the tree, and then walks back up updating value
// estimates along the way ("backpropagation" — a different, unrelated use
// of that word from neural network backpropagation!).
//
// Crucially: none of this ever touches the real environment. Every
// "imagined" state and afterstate here comes from the learned model.

const { softmaxArray, sampleCategorical, sampleDirichlet, MinMaxStats } = require('./utils');

class Node {
  constructor(prior) {
    this.prior = prior; // how likely the model thought this action/outcome was, before any search
    this.isChance = false; // set when the node is expanded
    this.visitCount = 0;
    this.valueSum = 0;
    this.reward = 0; // reward received when transitioning into this node (decision nodes only)
    this.state = null; // hidden state (decision node) or afterstate (chance node), as a plain array
    this.children = new Map(); // key: action index or chance-code index -> child Node
  }

  expanded() {
    return this.children.size > 0;
  }

  value() {
    if (this.visitCount === 0) return 0;
    return this.valueSum / this.visitCount;
  }
}

/** Turns raw policy logits into per-action priors, restricted to legal actions. */
function priorsForLegalActions(policyLogits, legalActions) {
  const legalLogits = legalActions.map((a) => policyLogits[a]);
  const legalProbs = softmaxArray(legalLogits);
  const priors = new Map();
  legalActions.forEach((action, i) => priors.set(action, legalProbs[i]));
  return priors;
}

/** Expands a decision node: run prediction f on its state, add one child per legal action. */
function expandDecisionNode(node, state, network, legalActions) {
  node.isChance = false;
  node.state = state;
  const { policyLogits, value } = network.predictionSingle(state);
  const priors = priorsForLegalActions(policyLogits, legalActions);
  for (const action of legalActions) {
    node.children.set(action, new Node(priors.get(action)));
  }
  return value;
}

/** Expands a chance node: run afterstate-prediction psi on its afterstate, add one child per possible chance code. */
function expandChanceNode(node, afterstate, network) {
  node.isChance = true;
  node.state = afterstate;
  const { chanceLogits, qValue } = network.afterstatePredictionSingle(afterstate);
  const priors = softmaxArray(chanceLogits);
  priors.forEach((prior, code) => {
    node.children.set(code, new Node(prior));
  });
  return qValue;
}

/** The UCB (upper confidence bound) score used to pick an action at a decision node. */
function ucbScore(config, minMaxStats, parent, child) {
  const pbC =
    Math.log((parent.visitCount + config.pbCBase + 1) / config.pbCBase) + config.pbCInit;
  const explorationTerm =
    (pbC * child.prior * Math.sqrt(parent.visitCount)) / (child.visitCount + 1);
  let exploitationTerm = 0;
  if (child.visitCount > 0) {
    exploitationTerm = minMaxStats.normalize(child.reward + config.discount * child.value());
  }
  return explorationTerm + exploitationTerm;
}

/** Chooses which child to descend into: best-UCB action for decision nodes, a sampled outcome for chance nodes. */
function selectChild(config, minMaxStats, node) {
  if (node.isChance) {
    const codes = [...node.children.keys()];
    const probs = codes.map((c) => node.children.get(c).prior);
    const chosen = sampleCategorical(codes, probs);
    return [chosen, node.children.get(chosen)];
  }

  let bestKey = null;
  let bestChild = null;
  let bestScore = -Infinity;
  for (const [key, child] of node.children.entries()) {
    const score = ucbScore(config, minMaxStats, node, child);
    if (score > bestScore) {
      bestScore = score;
      bestKey = key;
      bestChild = child;
    }
  }
  return [bestKey, bestChild];
}

/** Adds Dirichlet noise to the root's priors so self-play keeps exploring instead of always repeating itself. */
function addExplorationNoise(config, root) {
  const actions = [...root.children.keys()];
  if (actions.length === 0) return;
  const noise = sampleDirichlet(config.rootDirichletAlpha, actions.length);
  const frac = config.rootExplorationFraction;
  actions.forEach((action, i) => {
    const child = root.children.get(action);
    child.prior = child.prior * (1 - frac) + noise[i] * frac;
  });
}

/** Walks back up the path just explored, updating each node's running average value. */
function backpropagate(searchPath, value, discount, minMaxStats) {
  for (let i = searchPath.length - 1; i >= 0; i--) {
    const node = searchPath[i];
    node.valueSum += value;
    node.visitCount += 1;
    minMaxStats.update(node.value());
    // The reward earned arriving at `node`, plus the (discounted) value of
    // everything beyond it, is the value "seen" one step further back.
    value = node.reward + discount * value;
  }
}

/**
 * Runs a full Monte Carlo Tree Search starting from a real observation,
 * and returns the completed search tree's root node. The root's children
 * visit counts tell you which actions the search favors.
 *
 * @param {Object} config
 * @param {StochasticMuZeroNetwork} network
 * @param {number[]} rootObservation - the real, current observation
 * @param {number[]} legalActions - which actions are legal right now
 */
function runMCTS({ config, network, rootObservation, legalActions }) {
  const minMaxStats = new MinMaxStats();

  // Step 1: turn the real observation into a hidden state (the only time
  // the real environment's observation is used at all).
  const rootState = network.representationSingle(rootObservation);

  // Step 2: create the root as a decision node and expand it immediately.
  const root = new Node(0);
  expandDecisionNode(root, rootState, network, legalActions);
  addExplorationNoise(config, root);

  for (let simulation = 0; simulation < config.numSimulations; simulation++) {
    let node = root;
    const searchPath = [node];
    const edgeHistory = []; // the actions/chance-codes chosen while descending

    // Walk down through already-expanded nodes, alternating decision and
    // chance selection, until we fall off the edge of the known tree.
    while (node.expanded()) {
      const [key, child] = selectChild(config, minMaxStats, node);
      edgeHistory.push(key);
      node = child;
      searchPath.push(node);
    }

    // `node` is a brand-new leaf. Expand it using whichever model function
    // matches what kind of edge we just followed to get here.
    const parent = searchPath[searchPath.length - 2];
    const lastKey = edgeHistory[edgeHistory.length - 1];
    let value;

    if (parent.isChance) {
      // We just sampled a chance outcome from an afterstate -> use the
      // dynamics function g to find the resulting real hidden state.
      const { nextState, reward } = network.dynamicsSingle(parent.state, lastKey);
      node.reward = reward;
      value = expandDecisionNode(node, nextState, network, legalActions);
    } else {
      // We just chose an action from a state -> use the afterstate
      // dynamics function phi to find the resulting afterstate.
      const afterstate = network.afterstateDynamicsSingle(parent.state, lastKey);
      value = expandChanceNode(node, afterstate, network);
    }

    backpropagate(searchPath, value, config.discount, minMaxStats);
  }

  return root;
}

/** Turns a completed search's root visit counts into a policy vector the same length as numActions. */
function visitCountsToPolicy(root, numActions) {
  const policy = new Array(numActions).fill(0);
  let total = 0;
  for (const [action, child] of root.children.entries()) {
    policy[action] = child.visitCount;
    total += child.visitCount;
  }
  if (total > 0) {
    for (let i = 0; i < numActions; i++) policy[i] /= total;
  }
  return policy;
}

/**
 * Picks a real action to play based on the search's visit counts.
 * temperature = 0 always plays the most-visited action (greedy).
 * temperature = 1 samples proportionally to visit counts (exploratory).
 */
function selectAction(root, temperature) {
  const actions = [...root.children.keys()];
  const visitCounts = actions.map((a) => root.children.get(a).visitCount);

  if (temperature === 0) {
    let bestIndex = 0;
    for (let i = 1; i < visitCounts.length; i++) {
      if (visitCounts[i] > visitCounts[bestIndex]) bestIndex = i;
    }
    return actions[bestIndex];
  }

  const scaled = visitCounts.map((v) => Math.pow(v, 1 / temperature));
  const sum = scaled.reduce((a, b) => a + b, 0);
  const probs = scaled.map((v) => v / sum);
  return sampleCategorical(actions, probs);
}

module.exports = {
  Node,
  runMCTS,
  visitCountsToPolicy,
  selectAction,
};
