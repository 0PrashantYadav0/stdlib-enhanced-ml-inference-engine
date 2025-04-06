export default {
    mode: 'development',
    entry: './main.js',
    output: {
      filename: 'bundle.js',
      path: new URL('./dist', import.meta.url).pathname
    }
  };