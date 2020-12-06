
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

# Helper functions for images.

labelnames = dict(
    # https://www.cs.toronto.edu/~kriz/cifar.html
    cifar10=('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    # https://www.cs.toronto.edu/~kriz/cifar.html
    cifar100=('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')
)

def make_label_getter(dataset):
    """Returns a function converting label indices to names."""
    def getter(label):
        if dataset in labelnames:
            return labelnames[dataset][label]
        return f'label={label}'
    return getter

def show_img(img, ax=None, title=None):
    """Shows a single image."""
    if ax is None:
        ax = plt.gca()
    ax.imshow(img[...])
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)

def show_img_grid(imgs, titles, titles_2=None):
    """Shows a grid of images."""
    n = int(np.ceil(len(imgs)**.5))
    _, axs = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    if titles_2 is None:
        for i, (img, title) in enumerate(zip(imgs, titles)):
            img = (img + 1) / 2  # Denormalize
            show_img(img, axs[i // n][i % n], title)
    else:
        for i, (img, title, title_2) in enumerate(zip(imgs, titles, titles_2)):
            img = (img + 1) / 2  # Denormalize
            show_img(img, axs[i // n][i % n], f"{title} > {title_2}")



# Make sure tf does not allocate gpu memory.
tf.config.experimental.set_visible_devices([], 'GPU')

parser = flags.argparser(models.KNOWN_MODELS.keys(),
                        input_pipeline.DATASET_PRESETS.keys())

args = parser.parse_args()

# Overwrite args
# args.name = "ViT-B_16-cifar10_TEST_`date +%F_%H%M%S"
args.model = "ViT-B_16"
args.logdir = "/tmp/vit_logs"
args.dataset = "cifar10"
args.accum_steps = 8
args.batch = 32
args.batch_eval = 32
args.eval_every = 2000
args.progress_every = 100
args.shuffle_buffer = 2000
args.warmup_steps = 50
args.output = "./ViT-B_16.npz"
args.pretrained = "ViT-B_16_cifar10.npz"


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
# batch = next(iter(ds_train))
gen_train = iter(ds_train)

ds_test = input_pipeline.get_data(
    dataset=args.dataset,
    mode='test',
    repeats=1,
    batch_size=args.batch_eval,
    tfds_data_dir=args.tfds_data_dir,
    tfds_manual_dir=args.tfds_manual_dir,
    include_original=True)
logger.info(ds_test)
logger.info('=' * 50)


import pdb
# Build VisionTransformer architecture
model = models.KNOWN_MODELS[args.model]
VisionTransformer = model.partial(num_classes=dataset_info['num_classes'])
# VisionTransformer = model.partial(num_classes=dataset_info['num_classes'])
pretrained_path = os.path.join(args.vit_pretrained_dir, f'{args.pretrained}')
params = checkpoint.load(pretrained_path)
params['pre_logits'] = {}  # Need to restore empty leaf for Flax.

# _, params = VisionTransformer.init_by_shape(
#     jax.random.PRNGKey(0),
#     # Discard the "num_local_devices" dimension for initialization.
#     [(batch['image'].shape[1:], batch['image'].dtype.name)])
# logger.info(f"VisionTransformer INIT {VisionTransformer}")
# logger.info('=' * 50)

# if args.pretrained is not None:
#     try:
#         pretrained_path = os.path.join(args.vit_pretrained_dir, f'{args.pretrained}')
#         params = checkpoint.load_pretrained(
#             pretrained_path=pretrained_path,
#             init_params=params,
#             model_config=models.CONFIGS[args.model],
#             logger=logger)
#         params['pre_logits'] = {}
#     except:
#         logger.warning("pretrained not loaded, as error occured.")
# else:
#     logger.info("Pretrained not loaded, as --pretrained NOT SET.")
# logger.info('=' * 50)

# batch = next(iter(ds_test.as_numpy_iterator()))

# for i in range((args.batch - 1) // 9 + 1):

#     images, labels = batch['image'][0][i*9:i*9+9], batch['label'][0][i*9:i*9+9]
#     titles_train = map(make_label_getter(args.dataset), labels.argmax(axis=1))

#     logits = VisionTransformer.call(params, images)
#     preds = flax.nn.softmax(logits)
#     titles_test = map(make_label_getter(args.dataset), preds.argmax(axis=1))
#     show_img_grid(images, titles_train, titles_test)
#     plt.show()

#     print(preds)

ds_test_iter = iter(ds_test)

for test_batch in ds_test_iter:
    pass