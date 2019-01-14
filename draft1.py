
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pandas as pd
import numpy as np
import torch
import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import re


import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from sklearn.model_selection import train_test_split


# In[2]:


spaces = re.compile(' +')


def remove_first_space(x):
    """
    :param x: word
    :type x: str
    :return: word withou space in front
    :rtype: str
    """
    if x[0] == " ":
        return x[1:]
    else:
        return x


def simple_pre_process_text_df(data, key='text'):
    """
    :param data: data frame with the colum 'text'
    :type data: pd.DataFrame
    :param key: colum key
    :type key: str
    """

    data[key] = data[key].apply(lambda x: x.lower())
    data[key] = data[key].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x))) # noqa
    data[key] = data[key].apply(remove_first_space) # noqa remove space in the first position
    data[key] = data[key].apply((lambda x: spaces.sub(" ", x))) # noqa remove double spaces


# In[3]:


df = pd.read_csv("fixed_data/boolean4_train.csv")

simple_pre_process_text_df(df, key="sentence1")
simple_pre_process_text_df(df, key="sentence2")
simple_pre_process_text_df(df, key="and_A")
simple_pre_process_text_df(df, key="and_B")




df_train, df_dev = train_test_split(df, test_size=0.20, random_state=1)
print("df_train", df_train.shape)
print("df_dev", df_dev.shape)
df_train.head(3)


# In[4]:


toy = True

if toy:
    df_train = df_train.head(100)


# In[5]:


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


# In[6]:


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


# In[7]:


ex = InputExample(guid=0,
                  text_a=df.sentence1.values[0],
                  text_b=df.sentence2.values[0],
                  label=df.label.values[0])


# In[8]:


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


# In[9]:


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


# In[10]:


class ContraProcessor(DataProcessor):
    """Processor for the ContraQA data set."""
    
    def get_train_examples(self, data_df):
        """See base class."""
        return self._create_examples(data_df, "train")

    def get_dev_examples(self, data_df):
        """See base class."""
        return self._create_examples(data_df, "dev")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, data_df, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i in range(data_df.shape[0]):
            data_df.iloc[i]
            guid = str(i) + "_" + set_type
            text_a = data_df.iloc[i].sentence1
            text_b = data_df.iloc[i].sentence2
            label = data_df.iloc[i].label
            ex = InputExample(guid=guid,
                             text_a=text_a,
                             text_b=text_b,
                             label=label)
            examples.append(ex)
        return examples


# In[11]:


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


# In[12]:


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


# In[13]:


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


# In[14]:


device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
n_gpu = torch.cuda.device_count()


# In[15]:


# Bert pre-trained model selected in the list: 
Bert_pre_trained_models = ["bert-base-uncased",
                           "bert-large-uncased",
                           "bert-base-cased",
                           "bert-large-cased",
                           "bert-base-multilingual-uncased",
                           "bert-base-multilingual-cased"]


args = {"local_rank": -1,
        "fp16": False,
        "train_batch_size":32,
        "eval_batch_size":8,
        'learning_rate':5e-5,
        "num_train_epochs":1.0,
        "seed":42,
        'max_seq_length':128,
        "gradient_accumulation_steps":1,
        "loss_scale":0,
        "do_lower_case": False, 
        "do_train":False,
        "bert_model": Bert_pre_trained_models[0],
        "task_name":"ContraQA",
        "warmup_proportion":0.1}



logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    device, n_gpu, bool(args["local_rank"] != -1), args["fp16"]))


# In[16]:


args["train_batch_size"] = int(args["train_batch_size"] / args["gradient_accumulation_steps"])


# In[17]:


random.seed(args["seed"])
np.random.seed(args["seed"])
torch.manual_seed(args["seed"])


# In[18]:


task_name = args["task_name"].lower()


# In[19]:


processors = {"contraqa": ContraProcessor}

num_labels_task = {"contraqa": 2}


# In[20]:


processor = processors[task_name]()
num_labels = num_labels_task[task_name]
label_list = processor.get_labels()


# In[21]:


tokenizer = BertTokenizer.from_pretrained(args["bert_model"], do_lower_case=args["do_lower_case"])


# In[22]:


train_examples = processor.get_train_examples(df_train)


# In[23]:


num_train_steps = int(len(train_examples) / args["train_batch_size"] / args["gradient_accumulation_steps"] * args["num_train_epochs"])
print(num_train_steps)


# In[24]:


model = BertForSequenceClassification.from_pretrained(args["bert_model"],
          cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args["local_rank"]),
          num_labels = num_labels)


# In[25]:


model.to(device)


# In[26]:


param_optimizer = list(model.named_parameters())


# In[27]:


no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
t_total = num_train_steps


# ### Prepare optimizer

# In[28]:


if args["fp16"]:
    try:
        from apex.optimizers import FP16_Optimizer
        from apex.optimizers import FusedAdam
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    optimizer = FusedAdam(optimizer_grouped_parameters,
                          lr=args["learning_rate"],
                          bias_correction=False,
                          max_grad_norm=1.0)
    if args.loss_scale == 0:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    else:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

else:
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args["learning_rate"],
                         warmup=args["warmup_proportion"],
                         t_total=t_total)


# Prepare for training

# In[30]:


global_step = 0
nb_tr_steps = 0
tr_loss = 0
train_features = convert_examples_to_features(train_examples, label_list, args["max_seq_length"], tokenizer)
logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_examples))
logger.info("  Batch size = %d", args["train_batch_size"])
logger.info("  Num steps = %d", num_train_steps)
all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
if args["local_rank"] == -1:
    train_sampler = RandomSampler(train_data)
else:
    train_sampler = DistributedSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args["train_batch_size"])

model.train()

for _ in trange(int(args["num_train_epochs"]), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args["gradient_accumulation_steps"] > 1:
                    loss = loss / args["gradient_accumulation_steps"]

                if args["fp16"]:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args["gradient_accumulation_steps"] == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args["learning_rate"] * warmup_linear(global_step/t_total, args["warmup_proportion"])
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

