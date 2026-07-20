// run.js
//
// An end-to-end example: play 2048 against the agent's own Monte Carlo
// Tree Search, store games in the replay buffer, and periodically take a
// training step. Run it with:
//
//   npm install            (only needed once)
//   npm run example
//
// A word of caution: the paper trains this environment for millions of
// self-play steps on TPUs, with a much bigger network (256 hidden units,
// 10 residual blocks, 100 simulations per move) than is practical on a
// laptop. The config below matches the ENVIRONMENT exactly (observation
// size, action count, chance codebook size) but shrinks the NETWORK and
// SEARCH way down, so this finishes in a reasonable amount of time. Don't
// expect strong 2048 play out of this as configured — the point is to
// confirm self-play and training are wired together correctly. Turn the
// sizes back up (and let it run much longer) once you want real results.
//
// If you're debugging a setup problem, windyLineEnv.js is a much smaller,
// faster environment to isolate issues with.

const { createAgent } = require('../ai-src/index');
const { TwentyFortyEightEnv } = require('./twentyFortyEightEnv');

const agent = createAgent({
  // --- These three describe the 2048 ENVIRONMENT itself, matching
  //     twentyFortyEightEnv.js and the paper's own setup exactly ---
  observationSize: TwentyFortyEightEnv.observationSize('binary'), // 16 cells x 31 bits = 496
  numActions: 4, // up / right / down / left
  codebookSize: 32, // one chance code per (empty cell) x (new tile is 2 or 4)

  // --- Network/search size, shrunk way down from the paper's for a
  //     laptop-friendly demo (paper: hiddenSize 256, 10 residual blocks,
  //     100 simulations, batch size 1024, millions of training steps) ---
  hiddenSize: 256,
  numResidualBlocks: 10,
  numSimulations: 100,
  batchSize: 1024,
  numUnrollSteps: 1000,
  tdSteps: 1000,
});

const NUM_EPISODES = 300;
const TRAIN_STEPS_PER_EPISODE = 4;
const recentScores = [];

async function main() {
  for (let episode = 1; episode <= NUM_EPISODES; episode++) {
    const env = new TwentyFortyEightEnv({ encoding: 'binary' });
    const trajectory = agent.selfPlayer.playGame(env);
    agent.replayBuffer.saveGame(trajectory);

    // env.score is the game's final tally: the sum of every merged
    // tile's value over the whole episode, exactly matching the "episode
    // reward" the paper reports for 2048.
    recentScores.push(env.score);
    if (recentScores.length > 20) recentScores.shift();

    let averageLoss = null;
    if (agent.replayBuffer.size() >= 4) {
      let lossSum = 0;
      for (let i = 0; i < TRAIN_STEPS_PER_EPISODE; i++) {
        lossSum += agent.trainer.trainStep(agent.replayBuffer);
      }
      averageLoss = lossSum / TRAIN_STEPS_PER_EPISODE;
    }

    const avgScore = recentScores.reduce((a, b) => a + b, 0) / recentScores.length;
    const lossText = averageLoss === null ? 'n/a' : averageLoss.toFixed(4);
    console.log(
      `episode ${episode}/${NUM_EPISODES} | moves: ${trajectory.length} | score: ${env.score} | avg score (last 20): ${avgScore.toFixed(0)} | loss: ${lossText}`
    );
  }
}

main();
