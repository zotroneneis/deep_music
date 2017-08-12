import json
import os

import tensorflow as tf
import yaml

from models import make_model


def main():
    with open("config.yaml", 'r') as configfile:
        config = yaml.load(configfile)

    tf.logging.set_verbosity(config['general']['logging'])

    model_name = config['general']['model_name']
    tf.logging.info('Initializing the model: {}'.format(model_name))

    model = make_model(config)

    if config['train']:
        tf.logging.info('Training {}'.format(model_name))
        model.train()

    if config['generate']:
        tf.logging.info(
            'Generating new music with the trained {}'.format(model_name))
        model.generate()


if __name__ == '__main__':
    main()
