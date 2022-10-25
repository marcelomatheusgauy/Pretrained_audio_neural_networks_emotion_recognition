The file train_utils.py contains various training utility functions necessary for running the model. These functions will process the data in batches (load and apply transformations such as spectogram) and run the batches through the model for training. The models call the classes defined in the repository https://github.com/qiuqiangkong/audioset_tagging_cnn. In order to use the files in this folder it is necessary to add them to the folder from that repository as they use the functions within to define the models.

The files run_training.py and run_finetuning.py are examples of functions that can be created using the train_utils.py functions to train the model on the emotion recognition training dataset (run_training.py) and to finetune a pretrained model on the emotion recognition training dataset (run_finetuning.py)

The files run_training_main.py and run_finetuning_main.py are examples of scripts that use run_training.py (run_training_main.py) and run_finetuning.py (run_finetuning_main.py) to do the whole training loop with repetitions and save the models as well as the logs with the results.

The pretrained models of the PANNs can be found at https://zenodo.org/record/3987831#.Y1gz0r7MJkg. The speech emotion recognition refinement dataset is available through the website of the SE&R workshop (https://sites.google.com/view/ser2022)

For questions regarding how to use the code please contact: marcelomatheusgauy@gmail.com

