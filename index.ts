import * as tf from "@tensorflow/tfjs-node";
import sqlite3 from "sqlite3";
import { open } from "sqlite";

const DATABASE_PATH = "./boards.db";

// Train model
// What model?

// Evaluate model

(async () => {
  let model = tf.sequential();
  let path = tf.io.fileSystem("./model");
  await model.save(path);

  let db = await open({
    filename: DATABASE_PATH,
    driver: sqlite3.Database
  });

  let loadPath = tf.io.fileSystem("./model/model.json");
  let newModel = await tf.loadLayersModel(loadPath);
  console.log(newModel);
})();