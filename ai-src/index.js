// index.js
//
// The front door to this library. Import from here, plug in your own
// environment, and you have a working Stochastic MuZero agent.
//
// Minimal usage:
//
//   const { createAgent } = require('stochastic-muzero-tfjs');
//   const config = require('stochastic-muzero-tfjs/src/config');
//   // ...override config.observationSize / numActions / etc. for your game...
//
//   const agent = createAgent(config);
//
//   for (let episode = 0; episode < 1000; episode++) {
//     const trajectory = agent.selfPlayer.playGame(new MyEnvironment());
//     agent.replayBuffer.saveGame(trajectory);
//
//     if (agent.replayBuffer.size() >= config.batchSize) {
//       const loss = agent.trainer.trainStep(agent.replayBuffer);
//       console.log(`episode ${episode}, loss ${loss.toFixed(4)}`);
//     }
//   }

const defaultConfig = require('./config');
const { StochasticMuZeroNetwork } = require('./network');
const { SelfPlayActor } = require('./selfplay');
const { ReplayBuffer } = require('./replayBuffer');
const { Trainer } = require('./train');
const mcts = require('./mcts');
const losses = require('./losses');
const utils = require('./utils');

/**
 * Builds a complete, ready-to-use agent: a network, a self-play actor, a
 * replay buffer, and a trainer, all wired together with a shared config.
 * Override any fields of `config` (a copy of config.js's defaults) to
 * match your environment before calling this.
 */
function createAgent(config = {}) {
  const fullConfig = { ...defaultConfig, ...config };
  const network = new StochasticMuZeroNetwork(fullConfig);
  const selfPlayer = new SelfPlayActor(network, fullConfig);
  const replayBuffer = new ReplayBuffer(fullConfig);
  const trainer = new Trainer(network, fullConfig);

  return { config: fullConfig, network, selfPlayer, replayBuffer, trainer };
}

module.exports = {
  createAgent,
  defaultConfig,
  StochasticMuZeroNetwork,
  SelfPlayActor,
  ReplayBuffer,
  Trainer,
  mcts,
  losses,
  utils,
};
