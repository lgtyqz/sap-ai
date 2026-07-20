// run.js
//
// A minimal, end-to-end example: play games against the toy WindyLineEnv
// using the agent's own Monte Carlo Tree Search, store them in the replay
// buffer, and periodically take a training step. Run it with:
//
//   npm install            (only needed once)
//   npm run example
//
// You should see the average reward per game slowly climb from close to
// 0 (random play) towards +1 (reliably reaching the goal) as training
// progresses, and the loss generally trend downward.

const { createAgent } = require('../ai-src/index');
const { WindyLineEnv } = require('./windyLineEnv');

// Override just the settings that describe THIS environment; everything
// else uses the library's defaults from src/config.js.
const agent = createAgent({
  observationSize: 5, // one-hot position on a line of 5 cells
  numActions: 3, // left / stay / right
  codebookSize: 3, // wind: calm / left gust / right gust
  hiddenSize: 32,
  numResidualBlocks: 1,
  numSimulations: 20,
  batchSize: 16,
  numUnrollSteps: 3,
  tdSteps: 5,
});

const NUM_EPISODES = 300;
const TRAIN_STEPS_PER_EPISODE = 4;
const recentRewards = [];

async function main() {
  for (let episode = 1; episode <= NUM_EPISODES; episode++) {
    const env = new WindyLineEnv();
    const trajectory = agent.selfPlayer.playGame(env);
    agent.replayBuffer.saveGame(trajectory);

    // The final reward recorded in the trajectory tells us how the
    // episode ended: +1 reached the goal, -1 fell in the pit, 0 timed out.
    const finalReward = trajectory[trajectory.length - 1].reward;
    recentRewards.push(finalReward);
    if (recentRewards.length > 20) recentRewards.shift();

    let averageLoss = null;
    if (agent.replayBuffer.size() >= 4) {
      let lossSum = 0;
      for (let i = 0; i < TRAIN_STEPS_PER_EPISODE; i++) {
        lossSum += agent.trainer.trainStep(agent.replayBuffer);
      }
      averageLoss = lossSum / TRAIN_STEPS_PER_EPISODE;
    }

    if (episode % 10 === 0) {
      const avgReward = recentRewards.reduce((a, b) => a + b, 0) / recentRewards.length;
      const lossText = averageLoss === null ? 'n/a' : averageLoss.toFixed(4);
      console.log(
        `episode ${episode}/${NUM_EPISODES} | avg reward (last 20): ${avgReward.toFixed(2)} | loss: ${lossText}`
      );
    }
  }
}

main();
