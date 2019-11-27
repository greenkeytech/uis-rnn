# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for UIS-RNN."""

import random
import string
from functools import partial
from itertools import chain
from multiprocessing.dummy import Pool
from collections import defaultdict

import numpy as np
import torch
from torch import autograd


class Logger:
  """A class for printing logging information to screen."""

  def __init__(self, verbosity):
    self._verbosity = verbosity

  def print(self, level, message):
    """Print a message if level is not higher than verbosity.

    Args:
      level: the level of this message, smaller value means more important
      message: the message to be printed
    """
    if level <= self._verbosity:
      print(message)


def generate_random_string(length=6):
  """Generate a random string of upper case letters and digits.

  Args:
    length: length of the generated string

  Returns:
    the generated string
  """
  return ''.join([
      random.choice(string.ascii_uppercase + string.digits)
      for _ in range(length)])


def enforce_cluster_id_uniqueness(cluster_ids):
  """Enforce uniqueness of cluster id across sequences.

  Args:
    cluster_ids: a list of 1-dim list/numpy.ndarray of strings

  Returns:
    a new list with same length of cluster_ids

  Raises:
    TypeError: if cluster_ids or its element has wrong type
  """
  if not isinstance(cluster_ids, list):
    raise TypeError('cluster_ids must be a list')
  new_cluster_ids = []
  for cluster_id in cluster_ids:
    sequence_id = generate_random_string()
    if isinstance(cluster_id, np.ndarray):
      cluster_id = cluster_id.tolist()
    if not isinstance(cluster_id, list):
      raise TypeError('Elements of cluster_ids must be list or numpy.ndarray')
    new_cluster_id = ['_'.join([sequence_id, s]) for s in cluster_id]
    new_cluster_ids.append(new_cluster_id)
  return new_cluster_ids


def concatenate_training_data(train_sequences, train_cluster_ids,
                              enforce_uniqueness=True, shuffle=True):
  """Concatenate training data.

  Args:
    train_sequences: a list of 2-dim numpy arrays to be concatenated
    train_cluster_ids: a list of 1-dim list/numpy.ndarray of strings
    enforce_uniqueness: a boolean indicated whether we should enfore uniqueness
      to train_cluster_ids
    shuffle: whether to randomly shuffle input order

  Returns:
    concatenated_train_sequence: a 2-dim numpy array
    concatenated_train_cluster_id: a list of strings

  Raises:
    TypeError: if input has wrong type
    ValueError: if sizes/dimensions of input or their elements are incorrect
  """
  # check input
  if not isinstance(train_sequences, list) or not isinstance(
      train_cluster_ids, list):
    raise TypeError('train_sequences and train_cluster_ids must be lists')
  if len(train_sequences) != len(train_cluster_ids):
    raise ValueError(
        'train_sequences and train_cluster_ids must have same size')
  orig_seq_len = len(train_cluster_id)
  train_cluster_ids = [
      x.tolist() if isinstance(x, np.ndarray) else x
      for x in train_cluster_ids]
  global_observation_dim = None
  for i, (train_sequence, train_cluster_id) in enumerate(
      zip(train_sequences, train_cluster_ids)):
    try:
      train_length, observation_dim = train_sequence.shape
    except:
      print('Given and invalid sequence {train_sequence}, deleting it')
      del train_sequences[i]
      del train_cluster_ids[i]
      continue
    if i == 0:
      global_observation_dim = observation_dim
    elif global_observation_dim != observation_dim:
      raise ValueError(
          'train_sequences must have consistent observation dimension')
    if not isinstance(train_cluster_id, list):
      raise TypeError(
          'Elements of train_cluster_ids must be list or numpy.ndarray')
    if len(train_cluster_id) != train_length:
      raise ValueError(
          'Each train_sequence and its train_cluster_id must have same length')

  if orig_seq_len != len(train_cluster_ids):
    print(f'Samples removed from input: original number {orig_seq_len}, final number {len(train_cluster_ids)}')
  # enforce uniqueness
  if enforce_uniqueness:
    train_cluster_ids = enforce_cluster_id_uniqueness(train_cluster_ids)

  # random shuffle
  if shuffle:
    zipped_input = list(zip(train_sequences, train_cluster_ids))
    random.shuffle(zipped_input)
    train_sequences, train_cluster_ids = zip(*zipped_input)

  # concatenate
  concatenated_train_sequence = np.concatenate(train_sequences, axis=0)
  concatenated_train_cluster_id = [x for train_cluster_id in train_cluster_ids
                                   for x in train_cluster_id]
  return concatenated_train_sequence, concatenated_train_cluster_id

def group_by_consecutive_ids(input_sequence):
  segments = []
  if len(input_sequence) == 1:
    segments.append(input_sequence)
  else:
    prev = 0
    for i in range(len(input_sequence) - 1):
      if input_sequence[i + 1] != input_sequence[i] + 1:
        segments.append(input_sequence[prev:(i + 1)])
        prev = i + 1
      if i + 1 == len(input_sequence) - 1:
        segments.append(input_sequence[prev:])

  return segments

def make_resampled_index_array(segments, number_samples=1):
  sampled_index_sequences = [segments.copy() for _  in range(number_samples)]
  list(map(np.random.shuffle, sampled_index_sequences))
  return [np.array(list(chain(*_))) for _ in sampled_index_sequences]


def sample_permuted_segments(index_sequence, number_samples):
  """Sample sequences with permuted blocks.

  Args:
    index_sequence: (integer array, size: L)
      - subsequence index
      For example, index_sequence = [1,2,6,10,11,12].
    number_samples: (integer)

      - number of subsampled block-preserving permuted sequences.
      For example, number_samples = 5

  Returns:
    sampled_index_sequences: (a list of numpy arrays) - a list of subsampled
      block-preserving permuted sequences. For example,
    ```
    sampled_index_sequences =
    [[10,11,12,1,2,6],
     [6,1,2,10,11,12],
     [1,2,10,11,12,6],
     [6,1,2,10,11,12],
     [1,2,6,10,11,12]]
    ```
      The length of "sampled_index_sequences" is "number_samples".
  """
  # group segments by block
  segments = group_by_consecutive_ids(index_sequence)
  output = make_resampled_index_array(segments, number_samples)
  return output


def _subsample_sequences_for_resizing(sequence, cluster_indices, num_permutations=1):
  return_tuple = ([], [])
  idx_set = np.array(cluster_indices)
  # Create num_permutations shuffles of this speaker indices preserving
  # blocks (continuous segments of speech)
  sampled_idx_sets = sample_permuted_segments(idx_set, num_permutations)
  # Extract indices of sequence for the current speaker
  subsampled_sequences = list(map(lambda x: sequence[x, :], sampled_idx_sets))
  subsequence_length = [len(idx_set) + 1] * len(subsampled_sequences)
  return_tuple = (subsampled_sequences, subsequence_length)

  return return_tuple


def resize_sequence(sequence, cluster_id, num_permutations=None):
  """Resize sequences for packing and batching.

  Args:
    sequence: (real numpy matrix, size: seq_len*obs_size) - observed sequence
    cluster_id: (numpy vector, size: seq_len) - cluster indicator sequence
    num_permutations: int - Number of permutations per utterance sampled.

  Returns:
    sub_sequences: A list of numpy array, with observation vector from the same
      cluster in the same list.
    seq_lengths: The length of each cluster (+1).
  """

  max_workers = 20
  pool = Pool(processes=max_workers)

  # Collect indices for belonging to each speaker
  cluster_indices = defaultdict(list)
  for idx, val in enumerate(cluster_id):
    cluster_indices[val].append(idx)

  spkr_sequences = pool.map(partial(_subsample_sequences_for_resizing, sequence),
                   list(cluster_indices.values()))
  sub_sequences = []
  seq_lengths = []
  for spkr_tuple in spkr_sequences:
    sub_sequences += spkr_tuple[0]
    seq_lengths += spkr_tuple[1]

  pool.close()
  pool.join()
  return sub_sequences, seq_lengths


def pack_sequence(sub_sequences, seq_lengths, batch_size,
                  observation_dim, device, max_seq_len=4000):
  """Pack sequences for training.

  Args:
    sub_sequences: A list of numpy array, with observation vector from the same
      cluster in the same list.
    seq_lengths: The length of each cluster (+1).
    batch_size: int or None - Run batch learning if batch_size is None. Else,
      run online learning with specified batch size.
    observation_dim: int - dimension for observation vectors
    device: str - Your device. E.g., `cuda:0` or `cpu`.
    max_seq_len: int - maximum number of frames in a sample to avoid memory errors

  Returns:
    packed_rnn_input: (PackedSequence object) packed rnn input
    rnn_truth: ground truth
  """
  num_clusters = len(seq_lengths)
  sorted_seq_lengths = np.sort(seq_lengths)[::-1]
  permute_index = np.argsort(seq_lengths)[::-1]

  if batch_size:
    # TODO: consider sampling without replacement, would probably need a class
    # With a batch size choose random subset of data
    mini_batch = np.sort(np.random.choice(num_clusters, batch_size))
    batch_clusters = batch_size
    lengths_to_use = sorted_seq_lengths[mini_batch]
    get_index = lambda i: mini_batch[i]
  else:
    # Use the whole dataset
    batch_clusters = num_clusters
    # Allocate new memory because the [::-1] operation above causes errors
    lengths_to_use = sorted_seq_lengths.copy()
    get_index = lambda i: i

  if sorted_seq_lengths[get_index(0)] > max_seq_len:
    # Some samples are too large and cause CUDA memory errors
    batch_length = max_seq_len
  else:
    batch_length = sorted_seq_lengths[get_index(0)]

  # Initialize output to zeros
  rnn_input = np.zeros((batch_length,
                        batch_clusters,
                        observation_dim))

  # Populate the output with samples
  for i in range(batch_clusters):
    sample = sub_sequences[permute_index[get_index(i)]]
    sample_len = sample.shape[0]
    # Restrict size of sample if necessary
    if sample_len > max_seq_len:
      subset_start = np.random.randint(0, sample_len - max_seq_len)
      sample = sample[subset_start:subset_start + max_seq_len - 1]
      # Update length for packing
      lengths_to_use[i] = max_seq_len
    rnn_input[1:sample_len + 1, i, :] = sample

  rnn_input = autograd.Variable(
      torch.from_numpy(rnn_input).float()).to(device)
  packed_rnn_input = torch.nn.utils.rnn.pack_padded_sequence(
      rnn_input, lengths_to_use, batch_first=False)
  # Ground truth is the input shifted
  rnn_truth = rnn_input[1:, :, :]
  return packed_rnn_input, rnn_truth


def output_result(model_args, training_args, test_record):
  """Produce a string to summarize the experiment."""
  accuracy_array, _ = zip(*test_record)
  total_accuracy = np.mean(accuracy_array)
  output_string = """
Config:
  sigma_alpha: {}
  sigma_beta: {}
  crp_alpha: {}
  learning rate: {}
  regularization: {}
  batch size: {}

Performance:
  averaged accuracy: {:.6f}
  accuracy numbers for all testing sequences:
  """.strip().format(
      training_args.sigma_alpha,
      training_args.sigma_beta,
      model_args.crp_alpha,
      training_args.learning_rate,
      training_args.regularization_weight,
      training_args.batch_size,
      total_accuracy)
  for accuracy in accuracy_array:
    output_string += '\n    {:.6f}'.format(accuracy)
  output_string += '\n' + '=' * 80 + '\n'
  filename = 'layer_{}_{}_{:.1f}_result.txt'.format(
      model_args.rnn_hidden_size,
      model_args.rnn_depth, model_args.rnn_dropout)
  with open(filename, 'a') as file_object:
    file_object.write(output_string)
  return output_string


def estimate_transition_bias(cluster_ids, smooth=1):
  """Estimate the transition bias.

  Args:
    cluster_id: Either a list of cluster indicator sequences, or a single
      concatenated sequence. The former is strongly preferred, since the
      transition_bias estimated from the latter will be inaccurate.
    smooth: int or float - Smoothing coefficient, avoids -inf value in np.log
      in the case of a sequence with a single speaker and division by 0 in the
      case of empty sequences. Using a small value for smooth decreases the
      bias in the calculation of transition_bias but can also lead to underflow
      in some remote cases, larger values are safer but less accurate.

  Returns:
    bias: Flipping coin head probability.
    bias_denominator: The denominator of the bias, used for multiple calls to
      fit().
  """
  transit_num = smooth
  bias_denominator = 2 * smooth
  for cluster_id_seq in cluster_ids:
    for entry in range(len(cluster_id_seq) - 1):
      transit_num += (cluster_id_seq[entry] != cluster_id_seq[entry + 1])
      bias_denominator += 1
  bias = transit_num / bias_denominator
  return bias, bias_denominator
