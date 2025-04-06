import exp from '@stdlib/math-base-special-exp';

/**
 * Computes the softmax of an input vector.
 */
export function softmax(z) {
  const n = z.length;
  const out = new Float32Array(n);
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    if (z[i] > max) max = z[i];
  }
  let sum = 0;
  for (let i = 0; i < n; i++) {
    out[i] = exp(z[i] - max);
    sum += out[i];
  }
  for (let i = 0; i < n; i++) {
    out[i] /= sum;
  }
  return out;
}

/**
 * Returns the index of the maximum value in an array.
 */
export function argMax(arr) {
  let max = arr[0], idx = 0;
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > max) {
      max = arr[i];
      idx = i;
    }
  }
  return idx;
}
