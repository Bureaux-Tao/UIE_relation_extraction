import argparse
import os

import torch
from torch import cuda
from torch.utils.data import DataLoader

from bert4torch.snippets import sequence_padding
from model import uie_model, tokenizer
from train import IEDataset, SpanEvaluator
from utils import logger, IEMapDataset, get_relation_type_dict, unify_prompt_name, tqdm


class CategoryEvaluatorCategory(SpanEvaluator):
    def evaluate_category(self, dataloder, model, key):
        self.reset()
        pbar = tqdm(total = self.valid_len)
        for x_true, y_true in dataloder:
            start_prob, end_prob = model.predict(*x_true)
            start_ids, end_ids = y_true
            num_correct, num_infer, num_label = self.compute(start_prob, end_prob, start_ids, end_ids)
            pbar.update()
            self.update(num_correct, num_infer, num_label)
            precision, recall, f1 = self.accumulate()
            pbar.set_description(
                '{}:  '.format(key) + 'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
            )
        pbar.close()
        precision, recall, f1 = self.accumulate()
        return f1, precision, recall, int(self.num_correct_spans), int(self.num_infer_spans), int(self.num_label_spans)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def collate_fn(batch):
    """example: {title, prompt, content, result_list}
    """
    batch_token_ids, batch_token_type_ids, batch_start_ids, batch_end_ids = [], [], [], []
    for example in batch:
        token_ids, token_type_ids, offset_mapping = tokenizer.encode(example["prompt"], example["content"],
                                                                     maxlen = args.max_seq_len,
                                                                     return_offsets = 'transformers')
        bias = 0
        for index in range(len(offset_mapping)):
            if index == 0:
                continue
            mapping = offset_mapping[index]
            if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
                bias = index
            if mapping[0] == 0 and mapping[1] == 0:
                continue
            offset_mapping[index][0] += bias
            offset_mapping[index][1] += bias
        start_ids = [0 for _ in range(len(token_ids))]
        end_ids = [0 for _ in range(len(token_ids))]
        for item in example["result_list"]:
            start = map_offset(item["start"] + bias, offset_mapping)
            end = map_offset(item["end"] - 1 + bias, offset_mapping)
            start_ids[start] = 1.0
            end_ids[end] = 1.0
        
        batch_token_ids.append(token_ids)
        batch_token_type_ids.append(token_type_ids)
        batch_start_ids.append(start_ids)
        batch_end_ids.append(end_ids)
    
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype = torch.long, device = device)
    batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids), dtype = torch.long,
                                        device = device)
    batch_start_ids = torch.tensor(sequence_padding(batch_start_ids), dtype = torch.float, device = device)
    batch_end_ids = torch.tensor(sequence_padding(batch_end_ids), dtype = torch.float, device = device)
    return [batch_token_ids, batch_token_type_ids], [batch_start_ids, batch_end_ids]


def map_offset(ori_offset, offset_mapping):
    """map ori offset to token offset
    """
    for index, span in enumerate(offset_mapping):
        if span[0] <= ori_offset < span[1]:
            return index
    return -1


def do_eval():
    model = uie_model
    if args.model_path != "":
        if not os.path.exists(args.model_path):
            raise ValueError("Please input the correct path of saved weight path.")
        else:
            logger.info("Start loading weight from " + args.model_path)
            model.load_weights(args.model_path)
            logger.info("Loading weight finish.")
    if cuda.is_available():
        model = model.cuda()
    
    test_ds = IEDataset(args.test_path, tokenizer = tokenizer,
                        max_seq_len = args.max_seq_len)
    if args.no_neg:
        d = []
        for i in test_ds.dataset:
            if len(i['result_list']) != 0:
                d.append(i)
        test_ds.dataset = d
    
    class_dict = {}
    relation_data = []
    if args.category:
        for data in test_ds.dataset:
            class_name = unify_prompt_name(data['prompt'])
            # Only positive examples are evaluated in category mode
            # if len(data['result_list']) != 0:
            if args.no_neg:
                if len(data['result_list']) != 0:
                    if "的" not in data['prompt']:
                        class_dict.setdefault(class_name, []).append(data)
                    else:
                        relation_data.append((data['prompt'], data))
            else:
                if "的" not in data['prompt']:
                    class_dict.setdefault(class_name, []).append(data)
                else:
                    relation_data.append((data['prompt'], data))
        relation_type_dict = get_relation_type_dict(relation_data)
    else:
        class_dict["all_classes"] = test_ds

    logger.info("Entities Categories F1:")
    for key in class_dict.keys():
        if args.category:
            test_ds = IEMapDataset(class_dict[key], tokenizer = tokenizer,
                                   max_seq_len = args.max_seq_len)
        else:
            test_ds = class_dict[key]
        
        test_data_loader = DataLoader(test_ds, batch_size = args.batch_size, collate_fn = collate_fn)
        
        metric = CategoryEvaluatorCategory(test_data_loader, valid_len = len(test_ds) // args.batch_size)
        f1, precision, recall, num_correct_spans, num_infer_spans, num_label_spans = metric.evaluate_category(
            test_data_loader, model, key)
        logger.info("-----------------------------")
        logger.info("Class Name: %s" % key)
        logger.info("Evaluation Correct: %.0f | Infer: %.0f | Label: %.0f | Precision: %.5f | Recall: %.5f | F1: %.5f" %
                    (num_correct_spans, num_infer_spans, num_label_spans, precision, recall, f1))
    
    if args.category and len(relation_type_dict.keys()) != 0:
        logger.info("Relations Categories F1:")
        for key in relation_type_dict.keys():
            test_ds = IEMapDataset(relation_type_dict[key], tokenizer = tokenizer,
                                   max_seq_len = args.max_seq_len)
            test_data_loader = DataLoader(test_ds, batch_size = args.batch_size, collate_fn = collate_fn)
            metric = CategoryEvaluatorCategory(test_data_loader, valid_len = len(test_ds) // args.batch_size)
            try:
                f1, precision, recall, num_correct_spans, num_infer_spans, num_label_spans = metric.evaluate_category(
                    test_data_loader, model, key)
            except Exception as e:
                pass
            logger.info("-----------------------------")
            logger.info("Class Name: X的%s" % key)
            logger.info(
                "Evaluation Correct: %.0f | Infer: %.0f | Label: %.0f | Precision: %.5f | Recall: %.5f | F1: %.5f" %
                (num_correct_spans, num_infer_spans, num_label_spans, precision, recall, f1))


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-m", "--model_path", type = str, required = True,
                        help = "The path of saved model that you want to load.")
    parser.add_argument("-t", "--test_path", type = str, required = True,
                        help = "The path of test set.")
    parser.add_argument("-b", "--batch_size", type = int, default = 16,
                        help = "Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_len", type = int, default = 512,
                        help = "The maximum total input sequence length after tokenization.")
    parser.add_argument("--category", action = 'store_true',
                        help = "Precision, recall and F1 score are calculated for each class separately if this option is enabled.")
    parser.add_argument("--no_neg", action = 'store_true',
                        help = "Precision, recall and F1 score are calculated for test set with negative data.")
    
    args = parser.parse_args()
    # yapf: enable
    
    do_eval()
