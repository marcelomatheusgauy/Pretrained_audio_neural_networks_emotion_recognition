The file transformer_encoder.py contains the implementation of the transformer encoder modules. As mentioned in the paper it is based on the implementation found at: http://nlp.seas.harvard.edu/2018/04/03/attention.html. Details can be found in the paper

The file train_utils.py contains various training utility functions necessary for running the model. These functions will process the data in batches (load and apply transformations such as mfcc and noise insertion) and run the batches through the model for training.

The files run_training.py and run_finetuning.py are examples of functions that can be created using the train_utils.py functions to train the model on the emotion recognition training dataset (run_training.py) and to finetune a pretrained model on the emotion recognition training dataset (run_finetuning.py)

The files run_training_main.py and run_finetuning_main.py are examples of scripts that use run_training.py (run_training_main.py) and run_finetuning.py (run_finetuning_main.py) to do the whole training loop with repetitions and save the models as well as the logs with the results.

The data used in the paper to pretrain the models can be found at https://zenodo.org/record/6794924#.Y1f3577MJkg. The speech emotion recognition refinement dataset is available through the website of the SE&R workshop (https://sites.google.com/view/ser2022)

For questions regarding how to use the code please contact: marcelomatheusgauy@gmail.com
