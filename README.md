#BlechRNN

Pipeline to estimate neural firing rates using an autoregressive LSTM.

##Modules:
- 'get_data': Load data from a specified file and preprocess it.
- 'model': Define the LSTM model.
- 'train': Train the LSTM model and predict neural firing rates.
- 'output': Save the predicted neural firing rates to a specified file,
            save the model to a specified file, and generate plots

##Functionality roadmap:
- Ability to run prespecified model on a single dataset
- Batch processing of multiple datasets
- Hyperparameter optimization
- Remote execution
