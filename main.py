
import functools
import glob
import os
import time

from clu import metric_writers

import numpy as np

import jax
import jax.numpy as jnp

import flax
import flax.optim as optim
import flax.jax_utils as flax_utils

import tensorflow as tf

from vit_jax import checkpoint
from vit_jax import flags
from vit_jax import hyper
from vit_jax import logging
from vit_jax import input_pipeline
from vit_jax import models
from vit_jax import momentum_clip


# Make sure tf does not allocate gpu memory.
tf.config.experimental.set_visible_devices([], 'GPU')

parser = flags.argparser(models.KNOWN_MODELS.keys(),
                        input_pipeline.DATASET_PRESETS.keys())

args = parser.parse_args()

# python3 -m vit_jax.train --name ViT-B_16-cifar10_`date +%F_%H%M%S` 
# --model ViT-B_16 --logdir /tmp/vit_logs --dataset cifar10 
# --accum_steps 8 --batch 32 --batch_eval 32 --eval_every 2000 --progress_every 100 
# --shuffle_buffer=2000 --warmup_steps 50 --output=./ViT-B_16.npz  --pretrained ViT-B_16.npz

args.name = "ViT-B_16-cifar10_test"

logdir = os.path.join(args.logdir, args.name)
logger = logging.setup_logger(logdir)
logger.info(args)