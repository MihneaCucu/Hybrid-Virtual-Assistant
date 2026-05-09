Use guide:
preprocessing.py - does the preprocessing for the massive dataset. Results can be saved in a folder (currently in "massive_processed")
model_training.py - implements the model training pipeline. Dataset can either be loaded from pre-existing folder, or fetched from preprocessing.py using the preprocess_dataset function.
NLU_model.py - contains the code for the model proper. Necessary for both training and testing steps.
model_testing.py - testing pipeline. Requires model loaded from respective folder. User is prompted with a text box and their question is inputed into the model for intent classification and slot filling. Returns placeholder response. 
