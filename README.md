# drone-source-seperation-using-deep-clustering
a repository for our work submission in the CS236605 Deep Learning course, Technion, Israel.
by shai guendelman and or steiner, summer of 2019.

## About the repo:
The structure of the model is as follows:
- **dataset** directory you should download the dataset
- **demo_cache** has some samples for demonstration in the `methods_demo.ipynb`
- **experiment_output** directory contains some output that some of the experiments produce, like the saved checkpoints and the results
- **pb_bss** is copied from the repo at [here](https://github.com/fgnt/pb_bss), it implements some algorithms for blind source seperation on multichannel signals
- **rotor-speech-results** has some examples of the input and output of a trained model. as we can't upload the trained model to github we just produced the results
- `environment.yml` is the conda environment file
- `.gitignore` contains the ignore command to ignore everything that is inside of the **dataset** and **experiment_output** directories
- `requirments.txt` lists the packages used to run the experiment on a windows 10 computer, using python 3.7
- `rotor-speech_results.html` is an exprted jupyter notebook that illustrats the way thet the evaluation metrics are calculated
- `run_experiment.bat` is the script used to run the model on our PC's, it can be easly converted into a BASH script.
- `methods_demo.ipynb` is a notebook that explains how the model works, to use it you need some python packeges installed that are in the `requirments.txt`, all the other notebooks are for evaluating the different experiments

### The main part:

- **spatial_two_mics** - *This* is were most of the logic happens. most of the parts are copied from the repo [here](https://github.com/etzinis/unsupervised_spatial_dc) the sub-directories are orginized in the following way:
- **data_generator** contains the models that are used to preprocess the data for faster training, there are some versions of the code there, the important part there is `create_cached_dataset.py` that calls `my_dregon_dataset_creator_and_storage.py` to read the audio files, make the mask estimations on them and create the STFT representation of them so they are ready to used for training. also contains `source_position_generator.py` that randomly generates positions to create the mixtures.
- **data_loaders** here are spacific loaders to the data file structure, this knows how to load to python the raw data from the dataset according to the given file structure. it is used by the data_generator above.
- **dnn** this contains the loss function, the function that calculates the SDR, the training experiment we used in `experiments/my_experiment.py` the model class, and model evaluation in the **modules** sub-dir. The **utils** sub-dir contains some functions and `my_fast_dataset.py` that implements a pytorch dataloader for our data, and `model_logger.py` that has functions for saving and loading trained models.
- **labels_inference** contains the different methods to infer the labels from the mixtures, ground truth and duet(phase difference clustering)
- **utils** contains the mixture constructor that takes into account the positions of the sources in a room
- `config.py` has paths to some important directories, I set them to work if you follow the insturctions below, if you want to change them for different things try to change them there

## How to reproduce the experiment:
1. download the dataset from my google drive: [my dregon dataset](https://drive.google.com/file/d/1ryVIrp-w9aalGUq1sIPqPwMSbAZ_XqcP/view?usp=sharing) and unzip it with the name *MY_DREGON* into the **dataset** directory
2. after having the *MY_DREGON* folder inside of *dataset*, you need to create the mixtures with your costum combination. got to `spatial_two_mics/data_generator/create_cached_dataset.py` and set the wanted mixtures to be made and the quantities of train, test and evaluation samples, and run it. it might take a while... when it finishes it should give you paths to the datasets. they will probably be found in the `dataset` directory.
3. you should look at `run_experiment.bat` to see the different input parameter for the script.

For running the experiment you should activate your env, and 
```
python ./spatial_two_mics/dnn/experiments/my_experiment.py
```
with the wanted paramenters. **NOTE**: You need to give as input the directory of the preproccessed data set that was given to you in the previous step, without the 'train'/'test'/'val' in the end.

GOOD LUCK!