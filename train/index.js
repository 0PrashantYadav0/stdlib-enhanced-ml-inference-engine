import { loadIris } from '../data/index.js';
import { NeuralNetwork } from '../model/index.js';

async function trainAndEval() {
  const samples = await loadIris();
  const nn = new NeuralNetwork(4, 6, 3);

  nn.train(samples, 2000, 0.5);

  // Evaluate
  let correct = 0;
  for (const { features, label } of samples) {
    const pred = nn.predict(features);
    if (pred === label) correct++;
  }
  console.log(`Final accuracy: ${(correct / samples.length * 100).toFixed(2)}%`);
}

trainAndEval().catch(err => console.error(err));
