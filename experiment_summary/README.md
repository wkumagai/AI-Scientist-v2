# AI-Scientist-v2 Experiment Results

## Overview
This directory contains the results of the compositional regularization experiment run with AI-Scientist-v2 on April 8, 2025.


## Best Solution
The best solution is a Seq2Seq model with compositional regularization for the SCAN dataset. See `best_solution_df2d1d92c6f0407f9ccfb1dfca692348.py` for the implementation details.


## Research Approach
The experiment explored compositional generalization in neural networks using the SCAN dataset. See `idea.md` for the research approach details.


## Performance
The model showed improved performance with compositional regularization, with the loss decreasing from 1.3193 to 1.1049 over 3 epochs.


## Experiment Results

The experiment was conducted using the SCAN dataset with the 'simple' configuration. The model was trained for 3 epochs with the following results:

- Initial loss: 1.3193
- Final loss: 1.1049
- Regularization term: decreased from 8.6408 to 0.0001
- Compositional Generalization Accuracy: 13.75%


## Implementation Details

- Model: Seq2Seq with LSTM encoder-decoder
- Embedding dimension: 128
- Optimizer: Adam with learning rate 1e-3
- Loss: CrossEntropyLoss with compositional regularization
- Regularization factor: 0.1

