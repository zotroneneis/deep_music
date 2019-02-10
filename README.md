deepmusic
==============================

The main part of this project is based on exploring the ability of different recurrent Long Short-Term Memory architectures to generate novel monophonic melodies. All details about this part of the project can be found in the corresponding [report](https://github.com/zotroneneis/deep_music/blob/master/reports/report_deepmusic.pdf).

In addition, we trained and tested a variational autoencoder on the same task.

All code is written in Python and uses TensorFlow. 

Project authors: Anna-Lena Popkes, Pascal Wenker

Project Organization
------------

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── midis              <- Original MIDI files 
    │   ├── notesequences      <- Computed notesequence protocols 
    │   └── sequence_examples  <- Sequence examples used to train the model 
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Final latex report for the project 
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── main.py        <- Main method 
    │   ├── config.yaml    <- Config file, storing all network parameters 
    │   │
    │   ├── data           <- Scripts to transform MIDI files to notesequences and 
    │   │   │                 notesequences to sequence examples
    │   │   │                 
    │   │   ├── 01_create_notesequences 
    │   │   └── 02_create_sequenceExampes 
    │   │
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── basic_model.py
    │   │   ├── attention_model.py
    │   │
    │   ├── helper         <- Scripts that contain helper functions used by the model 
    │   │   ├── misc.py
    │   │   └── visualization.py
    │   │
    │   ├── scripts        <- Scripts to create exploratory and results oriented visualizations
    │       ├── create_debug_midis.py 
    │       ├── midi_to_melody.py
    │       └── tensorboardify.py 
    │   
    │   
    ├── vae                 <- Additional project training a variational autoencoder
    │   │                   to generate music
    │   │                   
    │   ├── models          <- Trained and serialized models, model predictions, 
    │   │   │               or model predictions
    │   │   ├── checkpoints 
    │   │   ├── generated_midis 
    │   │   └── tensorboard 
    │   │                   
    │   ├── src
    │       │                 
    │       ├── main.py      <- Main method 
    │       ├── config.yaml  <- Config file, storing all network parameters
    │       ├── models       <- Model definition and code to train the model
    │       └── helper       <- Helper functions used to train the model 
    └── 


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
