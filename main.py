
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

from utils.plt_helper import *

#@markdown Select whether you would like to store data in your personal drive.
#@markdown
#@markdown If you select **yes**, you will need to authorize Colab to access
#@markdown your personal drive
#@markdown
#@markdown If you select **no**, then any changes you make will diappear when
#@markdown this Colab's VM restarts after some time of inactivity...
use_gdrive = 'yes'  #@param ["yes", "no"]

if use_gdrive == 'yes':
  from google.colab import drive
  drive.mount('/gdrive')
  root = '/gdrive/My Drive/ViT_AdvAttack'
  import os
  if not os.path.isdir(root):
    os.mkdir(root)
  os.chdir(root)
  print(f'\nChanged CWD to "{root}"')
else:
  from IPython import display
  display.display(display.HTML(
      '<h1 style="color:red">CHANGES NOT PERSISTED</h1>'))


# Make sure tf does not allocate gpu memory.
tf.config.experimental.set_visible_devices([], 'GPU')

parser = flags.argparser(models.KNOWN_MODELS.keys(),
                        input_pipeline.DATASET_PRESETS.keys())

args = parser.parse_args()

# python3 -m vit_jax.train --name ViT-B_16-cifar10_`date +%F_%H%M%S` 
# --model ViT-B_16 --logdir /tmp/vit_logs --dataset cifar10 
# --accum_steps 8 --batch 32 --batch_eval 32 --eval_every 2000 --progress_every 100 
# --shuffle_buffer=2000 --warmup_steps 50 --output=./ViT-B_16.npz  --pretrained ViT-B_16.npz

logdir = os.path.join(args.logdir, args.name)
logger = logging.setup_logger(logdir)
for key, value in args.__dict__.items():
    logger.info(f"{key:>20s}: {value}")
logger.info('=' * 50)

logger.info(f'Available devices: {jax.devices()}')
logger.info('=' * 50)

dataset_info = input_pipeline.get_dataset_info(args.dataset, 'train')
ds_train = input_pipeline.get_data(
    dataset=args.dataset,
    mode='train',
    repeats=None,
    mixup_alpha=args.mixup_alpha,
    batch_size=args.batch,
    shuffle_buffer=args.shuffle_buffer,
    tfds_data_dir=args.tfds_data_dir,
    tfds_manual_dir=args.tfds_manual_dir,
    include_original=True,
    preprocess_train_dataset=True,
    no_shuffle=True)
batch = next(iter(ds_train))
gen_train = iter(ds_train)
logger.info('=' * 50)


import pdb
# Build VisionTransformer architecture
model = models.KNOWN_MODELS[args.model]
VisionTransformer = model.partial(num_classes=1000)
# VisionTransformer = model.partial(num_classes=dataset_info['num_classes'])
_, params = VisionTransformer.init_by_shape(
    jax.random.PRNGKey(0),
    # Discard the "num_local_devices" dimension for initialization.
    [(batch['image'].shape[1:], batch['image'].dtype.name)])
logger.info(f"VisionTransformer INIT {VisionTransformer}")
logger.info('=' * 50)

if args.pretrained is not None:
    try:
        pretrained_path = os.path.join(args.vit_pretrained_dir, f'{args.pretrained}')
        params = checkpoint.load_pretrained(
            pretrained_path=pretrained_path,
            init_params=params,
            model_config=models.CONFIGS[args.model],
            logger=logger)
        params['pre_logits'] = {}
    except:
        logger.warning("pretrained not loaded, as error occured.")
else:
    logger.info("Pretrained not loaded, as --pretrained NOT SET.")
logger.info('=' * 50)

images = batch["image"]
labels = batch["label"]

y = VisionTransformer.call(params, images[0])
print(y)
# for d in gen_train:
#     print(d)
#     ori = d["ori"][0][0]
#     image = d["image"][0][0]
#     show_2_image(ori, image, block=True)