// selfplay.js
//
// Ties the environment, the network, and MCTS together to actually play
// games and record what happened, producing the trajectories that get
// stored in the replay buffer and later used for training.
//
// Your environment needs to implement this small interface:
//
//   class MyEnvironment {
//     reset() -> observation (number[])
//       Starts a new episode and returns the first observation.
//
//     step(action) -> { observation, reward, done }
//       Applies `action`, returns the resulting observation, the reward
//       received, and whether the episode has now ended.
//
//     legalActions() -> number[]
//       Returns which action indices are legal right now.
//   }
//
// That's it — everything else (planning, learning) is handled for you.

const { runMCTS, visitCountsToPolicy, selectAction } = require('./mcts');

class SelfPlayActor {
  constructor(network, config) {
    this.network = network;
    this.config = config;
  }

  /**
   * Plays one full episode against `env`, using MCTS to choose every
   * move, and returns the recorded trajectory in the shape the replay
   * buffer expects.
   *
   * @param {number} temperature - controls how greedily actions are
   *   chosen from the search's visit counts (see mcts.js selectAction).
   *   Pass a smaller number (or a function of training progress) as the
   *   agent gets stronger, if you want less randomness over time.
   */
  playGame(env, temperature = this.config.actionSelectionTemperature) {
    const trajectory = [];
    let observation = env.reset();
    let done = false;
    let pendingReward = 0; // reward received arriving at the CURRENT observation

    while (!done) {
      const legalActions = env.legalActions();

      const root = runMCTS({
        config: this.config,
        network: this.network,
        rootObservation: observation,
        legalActions,
      });

      const searchPolicy = visitCountsToPolicy(root, this.config.numActions);
      const searchValue = root.value();
      const action = selectAction(root, temperature);

      trajectory.push({
        observation,
        action,
        reward: pendingReward,
        searchPolicy,
        searchValue,
      });

      const step = env.step(action);
      observation = step.observation;
      pendingReward = step.reward;
      done = step.done;
    }

    // Record the final observation too (with reward but no further move),
    // so value targets near the end of the game have something real to
    // bootstrap from instead of running off the end of the array.
    trajectory.push({
      observation,
      action: 0, // never actually used past this point
      reward: pendingReward,
      searchPolicy: new Array(this.config.numActions).fill(1 / this.config.numActions),
      searchValue: 0, // the game is over, so there is no more future value
    });

    return trajectory;
  }
}

module.exports = { SelfPlayActor };
