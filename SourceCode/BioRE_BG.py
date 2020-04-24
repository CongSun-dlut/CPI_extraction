# coding=utf-8
# Author: Cong Sun
# Email:  suncong132@mail.dlut.edu.cn
#
# BioRE_BG
#
#--------------------------------------------------------------------------



from __future__ import absolute_import, division, print_function
import argparse
import csv
import logging
import os
import random
import sys
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss
import scipy
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, precision_recall_fscore_support

import modeling_BG
from modeling_BG import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
import os

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

sigma = 2.5
mu = 0.0

norm_0=scipy.stats.norm(mu,sigma).cdf(1)-scipy.stats.norm(mu,sigma).cdf(0)
norm_1=scipy.stats.norm(mu,sigma).cdf(2)-scipy.stats.norm(mu,sigma).cdf(1)
norm_2=scipy.stats.norm(mu,sigma).cdf(3)-scipy.stats.norm(mu,sigma).cdf(2)
norm_3=scipy.stats.norm(mu,sigma).cdf(4)-scipy.stats.norm(mu,sigma).cdf(3)
norm_4=scipy.stats.norm(mu,sigma).cdf(5)-scipy.stats.norm(mu,sigma).cdf(4)
norm_5=scipy.stats.norm(mu,sigma).cdf(6)-scipy.stats.norm(mu,sigma).cdf(5)
norm_6=scipy.stats.norm(mu,sigma).cdf(7)-scipy.stats.norm(mu,sigma).cdf(6)



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample."""
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, P_gauss1_list, P_gauss2_list, label_id):
        
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        
        self.P_gauss1_list = P_gauss1_list
        self.P_gauss2_list = P_gauss2_list
        self.label_id = label_id


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

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class BioBERTChemprotProcessor(DataProcessor):
    """Processor for the BioBERT data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")


    def get_labels(self):
        """See base class."""
        return ["CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9", "false"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if set_type == "train" and i == 0:
                continue
            if set_type == "dev" and i == 0:
                continue
            if set_type == "test" and i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            # title = line[1]
            text_a = line[2]
            # sdp = line[3]
            label = line[4]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class BioBERT_DDIProcessor(DataProcessor):
    """Processor for the BioBERT data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["DDI-advise", "DDI-effect", "DDI-mechanism", "DDI-int", "DDI-false"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if set_type == "train" and i == 0:
                continue
            if set_type == "dev" and i == 0:
                continue
            if set_type == "test" and i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

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

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        if '@' not in tokens:
            P_gauss1_array = np.zeros(max_seq_length)
            P_gauss2_array = np.zeros(max_seq_length)
        else:
            ann1_index=tokens.index('@')
            if '@' not in tokens[ann1_index+1:]:
                P_array = np.zeros(max_seq_length)
                P_array[ann1_index-1] = norm_1
                P_array[ann1_index] = norm_0
                P_array[ann1_index+1] = norm_0
                if ann1_index+2 < max_seq_length:
                    P_array[ann1_index+2] = norm_1
                if ann1_index+3 < max_seq_length:
                    P_array[ann1_index+3] = norm_2
                if ann1_index+4 < max_seq_length:
                    P_array[ann1_index+4] = norm_3
                if ann1_index+5 < max_seq_length:
                    P_array[ann1_index+5] = norm_4
                if ann1_index+6 < max_seq_length:
                    P_array[ann1_index+6] = norm_5
                if ann1_index+7 < max_seq_length:
                    P_array[ann1_index+7] = norm_6
                if ann1_index-2 >= 0:
                    P_array[ann1_index-2] = norm_2
                if ann1_index-3 >= 0:
                    P_array[ann1_index-3] = norm_3
                if ann1_index-4 >= 0:
                    P_array[ann1_index-4] = norm_4
                if ann1_index-5 >= 0:
                    P_array[ann1_index-5] = norm_5
                if ann1_index-6 >= 0:
                    P_array[ann1_index-6] = norm_6
                P_gauss1_array = P_array
                P_gauss2_array = P_array
            else:
                ann2_index=tokens.index('@',ann1_index+1)
                P1_array = np.zeros(max_seq_length)
                P2_array = np.zeros(max_seq_length)
    
                P1_array[ann1_index-1] = norm_1
                P1_array[ann1_index] = norm_0
                P1_array[ann1_index+1] = norm_0
                if ann1_index+2 < max_seq_length:
                    P1_array[ann1_index+2] = norm_1
                if ann1_index+3 < max_seq_length:
                    P1_array[ann1_index+3] = norm_2
                if ann1_index+4 < max_seq_length:
                    P1_array[ann1_index+4] = norm_3
                if ann1_index+5 < max_seq_length:
                    P1_array[ann1_index+5] = norm_4
                if ann1_index+6 < max_seq_length:
                    P1_array[ann1_index+6] = norm_5
                if ann1_index+7 < max_seq_length:
                    P1_array[ann1_index+7] = norm_6
                if ann1_index-2 >=0:
                    P1_array[ann1_index-2] = norm_2
                if ann1_index-3 >=0:
                    P1_array[ann1_index-3] = norm_3
                if ann1_index-4 >=0:
                    P1_array[ann1_index-4] = norm_4
                if ann1_index-5 >=0:
                    P1_array[ann1_index-5] = norm_5
                if ann1_index-6 >=0:
                    P1_array[ann1_index-6] = norm_6
                    
                P2_array[ann2_index-1] = norm_1
                P2_array[ann2_index] = norm_0
                P2_array[ann2_index+1] = norm_0
                if ann2_index+2 < max_seq_length:
                    P2_array[ann2_index+2] = norm_1
                if ann2_index+3 < max_seq_length:
                    P2_array[ann2_index+3] = norm_2
                if ann2_index+4 < max_seq_length:
                    P2_array[ann2_index+4] = norm_3
                if ann2_index+5 < max_seq_length:
                    P2_array[ann2_index+5] = norm_4
                if ann2_index+6 < max_seq_length:
                    P2_array[ann2_index+6] = norm_5
                if ann2_index+7 < max_seq_length:
                    P2_array[ann2_index+7] = norm_6
                if ann2_index-2 >=0:
                    P2_array[ann2_index-2] = norm_2
                if ann2_index-3 >= 0:
                    P2_array[ann2_index-3] = norm_3
                if ann2_index-4 >= 0:
                    P2_array[ann2_index-4] = norm_4
                if ann2_index-5 >= 0:
                    P2_array[ann2_index-5] = norm_5
                if ann2_index-6 >= 0:
                    P2_array[ann2_index-6] = norm_6
                P_gauss1_array = P1_array 
                P_gauss2_array = P2_array

        P_gauss1_list= list(P_gauss1_array)
        P_gauss2_list= list(P_gauss2_array)
        

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
        assert len(P_gauss1_list) == max_seq_length
        assert len(P_gauss2_list) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))            
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))            
            logger.info("P_gauss1_list: %s" % " ".join([str(x) for x in P_gauss1_list]))
            logger.info("P_gauss2_list: %s" % " ".join([str(x) for x in P_gauss2_list]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              P_gauss1_list=P_gauss1_list,
                              P_gauss2_list=P_gauss2_list,
                              label_id=label_id))
    return features


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

def f1_chemprot(preds, labels):
    p,r,f,s = precision_recall_fscore_support(y_pred=preds, y_true=labels, labels=[0,1,2,3,4], average="micro")
    return {
        "P": p,
        "R": r,
        "F1": f,
    }

def f1_ddi(preds, labels):
    p,r,f,s = precision_recall_fscore_support(y_pred=preds, y_true=labels, labels=[0,1,2,3], average="micro")
    return {
        "P": p,
        "R": r,
        "F1": f,
    }

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cpi":
        return f1_chemprot(preds, labels)
    elif task_name == "ddi":
        return f1_ddi(preds, labels)
    else:
        raise KeyError(task_name)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--saved_model", default=None, type=str,
                        help="saved model")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run test on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--predict_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for predict.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=2.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=47,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "cpi": BioBERTChemprotProcessor,
        "ddi": BioBERT_DDIProcessor,
    }

    output_modes = {
        "cpi": "classification",
        "ddi": "classification",
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case, 
                never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9"))

    tokenizer.vocab["CPR:3"] = 3
    tokenizer.vocab["CPR:4"] = 4
    tokenizer.vocab["CPR:5"] = 5
    tokenizer.vocab["CPR:6"] = 6
    tokenizer.vocab["CPR:9"] = 9
    train_examples = None
    num_train_optimization_steps = None
    

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
              cache_dir=cache_dir,
              num_labels=num_labels)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps,
                             e=1e-08)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_P_gauss1_list = torch.tensor([f.P_gauss1_list for f in train_features], dtype=torch.float)
        all_P_gauss2_list = torch.tensor([f.P_gauss2_list for f in train_features], dtype=torch.float)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_P_gauss1_list, all_P_gauss2_list,all_label_ids)
        
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, P_gauss1_list, P_gauss2_list, label_ids = batch

                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, P_gauss1_list, P_gauss2_list, labels=None)

                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

    else:
        output_model_file = os.path.join(args.saved_model, 'pytorch_model.bin')
        output_config_file = os.path.join(args.saved_model, 'bert_config.json')
        config = BertConfig(output_config_file)
        model = BertForSequenceClassification(config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))
        model.to(device)


    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_P_gauss1_list = torch.tensor([f.P_gauss1_list for f in eval_features], dtype=torch.float)
        all_P_gauss2_list = torch.tensor([f.P_gauss2_list for f in eval_features], dtype=torch.float)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_P_gauss1_list, all_P_gauss2_list, all_label_ids)  
        
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for input_ids, input_mask, segment_ids, P_gauss1_list, P_gauss2_list, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            P_gauss1_list = P_gauss1_list.to(device)
            P_gauss2_list = P_gauss2_list.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, P_gauss1_list, P_gauss2_list, labels=None)

            # create eval loss and other metric required by the task
            if output_mode == "classification":
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            elif output_mode == "regression":
                loss_fct = MSELoss()
                tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
            
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(task_name, preds, all_label_ids.numpy())
        loss = tr_loss/nb_tr_steps if args.do_train else None

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

                
    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running prediction *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.predict_batch_size)
        
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_P_gauss1_list = torch.tensor([f.P_gauss1_list for f in test_features], dtype=torch.float)
        all_P_gauss2_list = torch.tensor([f.P_gauss2_list for f in test_features], dtype=torch.float)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.float)

        predict_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_P_gauss1_list, all_P_gauss2_list, all_label_ids)

        # Run prediction for full data
        predict_sampler = SequentialSampler(predict_data)
        predict_dataloader = DataLoader(predict_data, sampler=predict_sampler, batch_size=args.predict_batch_size)

        model.eval()
        test_loss = 0
        nb_test_steps = 0
        preds = []

        for input_ids, input_mask, segment_ids, P_gauss1_list, P_gauss2_list,label_ids in tqdm(predict_dataloader, desc="Predicting"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            P_gauss1_list = P_gauss1_list.to(device)
            P_gauss2_list = P_gauss2_list.to(device)
            label_ids = label_ids.to(device)
            

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, P_gauss1_list, P_gauss2_list, labels=None)

            # create eval loss and other metric required by the task
            if output_mode == "classification":
                loss_fct = CrossEntropyLoss()
                tmp_test_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        
            elif output_mode == "regression":
                loss_fct = MSELoss()
                tmp_test_loss = loss_fct(logits.view(-1), label_ids.view(-1))
            
            test_loss += tmp_test_loss.mean().item()
            nb_test_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        test_loss = test_loss / nb_test_steps
        preds = preds[0]
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(task_name, preds, all_label_ids.numpy())
        loss = tr_loss/nb_tr_steps if args.do_train else None

        result['test_loss'] = test_loss
        result['global_step'] = global_step
        result['loss'] = loss

        output_test_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_file, "w") as writer:
            logger.info("***** Predict results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()