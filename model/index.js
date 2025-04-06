import randn from '@stdlib/random-base-randn';
import { softmax } from '../utils/index.js';

export class NeuralNetwork {
  constructor(inputSize, hiddenSize, outputSize) {
    this.inputSize  = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;

    this.W1 = this._randMatrix(inputSize, hiddenSize);
    this.W2 = this._randMatrix(hiddenSize, outputSize);
  }

  _randMatrix(rows, cols) {
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => randn() * 0.1)
    );
  }

  _dotRowVec(vec, mat) {
    return mat[0].map((_, j) =>
      vec.reduce((sum, v, i) => sum + v * mat[i][j], 0)
    );
  }

  _sigmoid(arr) {
    return arr.map(v => 1 / (1 + Math.exp(-v)));
  }

  _sigmoidDeriv(arr) {
    return arr.map(v => v * (1 - v));
  }

  _subtract(A, B) {
    return A.map((row, i) =>
      row.map((v, j) => v - B[i][j])
    );
  }

  _scale(A, α) {
    return A.map(row => row.map(v => v * α));
  }

  /**
   * Forward pass for one sample.
   */
  forward(inputVec) {
    const Z1 = this._dotRowVec(inputVec, this.W1);
    const A1 = this._sigmoid(Z1);

    const Z2 = this._dotRowVec(A1, this.W2);
    const probs = softmax(new Float32Array(Z2));

    return { hidden: A1, probabilities: probs };
  }

  /**
   * Train with vanilla SGD and cross-entropy loss.
   */
  train(samples, epochs = 100, lr = 0.1) {
    for (let e = 0; e < epochs; e++) {
      let epochLoss = 0;

      for (const { features, label } of samples) {
        const { hidden: A1, probabilities: P } = this.forward(features);

        const T = Array(this.outputSize).fill(0);
        T[label] = 1;

        const errorOut = P.map((p, i) => p - T[i]);
        const dW2 = A1.map(h => errorOut.map(err => h * err));

        const hiddenErr = this.W2.map((row, i) =>
          row.reduce((sum, w, j) => sum + w * errorOut[j], 0)
        );
        const delta1 = this._sigmoidDeriv(A1).map((d, i) => d * hiddenErr[i]);

        const dW1 = Array.from({ length: this.inputSize }, (_, i) =>
          delta1.map(d => features[i] * d)
        );

        this.W2 = this._subtract(this.W2, this._scale(dW2, lr));
        this.W1 = this._subtract(this.W1, this._scale(dW1, lr));

        epochLoss += -T.reduce((sum, t, i) =>
          sum + t * Math.log(P[i] + 1e-15), 0
        );
      }
    }
  }

  /**
   * Predict class label for a sample.
   */
  predict(features) {
    const { probabilities } = this.forward(features);
    return probabilities.indexOf(Math.max(...probabilities));
  }
}
