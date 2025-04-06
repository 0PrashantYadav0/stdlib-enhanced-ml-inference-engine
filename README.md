# Enhanced Machine Learning Inference Engine

## Project Overview

This project is a JavaScript-based machine learning inference engine that implements a neural network for classifying the Iris dataset. Built using the [stdlib](https://stdlib.io/) library for mathematical operations, it demonstrates how to create and train a neural network entirely in JavaScript with efficient numerical computation capabilities.

The project features a simple but effective neural network architecture with one hidden layer, which achieves high accuracy on the classic Iris flower classification problem. It showcases both browser-based and Node.js execution environments.

## Table of Contents

- [Project Overview](#project-overview)
- [Detailed Description](#detailed-description)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [License](#license)

## Detailed Description

The Enhanced Machine Learning Inference Engine consists of the following key components:

1. **Neural Network Model**:
   - Implemented in [`model/index.js`](model/index.js )
   - Features a fully-connected architecture with one hidden layer
   - Uses sigmoid activation for the hidden layer and softmax for output probabilities
   - Includes forward propagation and backpropagation with SGD (Stochastic Gradient Descent)

2. **Data Processing**:
   - Manages the Iris dataset in [`data/index.js`](data/index.js )
   - Performs feature normalization using min-max scaling
   - Converts categorical targets to numeric labels

3. **Utility Functions**:
   - Provides mathematical operations in [`utils/index.js`](utils/index.js )
   - Implements the softmax activation function
   - Includes helper functions like argMax for prediction

4. **Web Interface**:
   - Offers an interactive browser-based demo in [`index.html`](index.html )
   - Displays model accuracy before and after training
   - Shows predictions on sample data points

5. **Training Script**:
   - Provides a standalone training process in [`train/index.js`](train/index.js )
   - Allows for model evaluation in a Node.js environment

## Setup Instructions

### Prerequisites

- [Node.js](https://nodejs.org/) (v14.x or later recommended)
- [npm](https://www.npmjs.com/) (included with Node.js)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/enhanced-ml-inference-engine.git
   cd enhanced-ml-inference-engine
   ```

2. **Install dependencies**:

   ```bash
   npm install
   ```

3. **Build the project**:

   ```bash
   npm run build
   ```

   This will use webpack to bundle the application into the [`dist`](dist ) directory.

### Running the Application

#### Web Interface

1. **Start the development server**:

   ```bash
   npm start
   ```

2. **Access the application**:
   Open your web browser and navigate to `http://localhost:8080`

3. **View the results**:
   The interface will show the neural network's performance on the Iris dataset.

#### Training Script

To run the training script directly (useful for development and testing):

```bash
npm run train
```

This will train the model on the Iris dataset and report the final accuracy in the terminal.

## License

This project is licensed under the MIT License:

```license
MIT License

Copyright (c) 2025 Enhanced Machine Learning Inference Engine

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
