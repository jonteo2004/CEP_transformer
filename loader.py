import tensorflow as tf
import argparse
from Main import evaluate
from dataset import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument(
    '--max_samples',
    default=25000,
    type=int,
    help='maximum number of conversation pairs to use')
parser.add_argument(
    '--max_length', default=40, type=int, help='maximum sentence length')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_layers', default=2, type=int)
parser.add_argument('--num_units', default=512, type=int)
parser.add_argument('--d_model', default=256, type=int)
parser.add_argument('--num_heads', default=8, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--activation', default='relu', type=str)
parser.add_argument('--epochs', default=20, type=int)

hparams = parser.parse_args()
dataset, tokenizer = get_dataset(hparams)

model = tf.keras.models.load_model('saved_model\d_1024\\b_128\E_440')
model.summary()
evaluate(hparams, model, tokenizer)