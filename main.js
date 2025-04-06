import { loadIris } from './data/index.js';
import { NeuralNetwork } from './model/index.js';

async function main() {
  const nn = new NeuralNetwork(4, 6, 3);
  const samples = await loadIris();
  const resultsEl = document.getElementById('results');

  // Initial accuracy
  let correct0 = 0;
  for (const { features, label } of samples) {
    if (nn.predict(features) === label) correct0++;
  }
  resultsEl.innerHTML = `<p>Initial accuracy: ${(correct0 / samples.length * 100).toFixed(2)}%</p>`;

  nn.train(samples, 2000, 0.5);

  // Post-training accuracy
  let correct1 = 0;
  let html = '<div class="predictions">';
  samples.forEach((s, i) => {
    const pred = nn.predict(s.features);
    const ok   = pred === s.label;
    if (ok) correct1++;
    if (i < 5) {
      const { probabilities } = nn.forward(s.features);
      html += `
        <div class="prediction ${ok ? 'correct' : 'incorrect'}">
          <strong>Sample ${i+1}:</strong>
          True=${s.label}, Pred=${pred}
          <span class="probs">[${probabilities.map(p => p.toFixed(2)).join(', ')}]</span>
        </div>
      `;
    }
  });
  html += '</div>';

  resultsEl.innerHTML += `
    <p>Post‑training accuracy: ${(correct1 / samples.length * 100).toFixed(2)}%</p>
    <p>First 5 predictions:</p>
    ${html}
  `;
}

main().catch(err => {
  document.getElementById('results').innerHTML = `<p class="error">Error: ${err.message}</p>`;
});
