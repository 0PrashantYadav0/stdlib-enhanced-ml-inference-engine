/**
 * Loads the Iris dataset from a local JSON file.
 * @returns {Promise<Array<{features: Float32Array, label: number}>>}
 */
export async function loadIris() {
  try {
    const response = await fetch('data/iris.json');
    const { data: features, target } = await response.json();

    // Normalize features (min–max per column)
    const transpose = m => m[0].map((_, i) => m.map(row => row[i]));
    const t = transpose(features);
    const tNorm = t.map(col => {
      const min = Math.min(...col), max = Math.max(...col);
      return col.map(v => (v - min) / (max - min));
    });
    const X = transpose(tNorm).map(row => new Float32Array(row));

    return X.map((featuresRow, i) => ({
      features: featuresRow,
      label: target[i]
    }));
  } catch (err) {
    throw err;
  }
}
