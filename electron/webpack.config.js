module.exports = {
  mode: 'development',
  entry: './src/performanceRNN.ts',
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: 'ts-loader',
      }
    ]
  },
  // import 文で .ts ファイルを解決するため
  resolve: {
    extensions: [
      '.ts',
      '.js'
    ]
  }
};