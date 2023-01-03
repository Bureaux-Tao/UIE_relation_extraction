import argparse
import os
import ipykernel
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bert4torch.losses import MultilabelCategoricalCrossentropy, QFocalLoss
from bert4torch.optimizers import extend_with_exponential_moving_average, get_linear_schedule_with_warmup
from model import uie_model, tokenizer
from bert4torch.snippets import seed_everything, sequence_padding, Callback, EarlyStopping, Logger, Tensorboard
from torch import nn
from torch.utils.data import Dataset
from torchinfo import summary
import json
from utils import get_bool_ids_greater_than, get_span, tqdm, logger
from random import sample

seed_everything(42)
history = {"epoch": [], "loss": [], "f1": [], "precision": [], "recall": []}
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class IEDataset(Dataset):
    """信息抽取
    """
    
    def __init__(self, file_path, tokenizer, max_seq_len, fewshot = None) -> None:
        super().__init__()
        self.file_path = file_path
        if fewshot is None:
            self.dataset = list(self.reader(file_path))
        else:
            assert isinstance(fewshot, int)
            self.dataset = sample(list(self.reader(file_path)), fewshot)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]
    
    @staticmethod
    def reader(data_path, max_seq_len = 512):
        """read json
        """
        with open(data_path, 'r', encoding = 'utf-8') as f:
            for line in f:
                json_line = json.loads(line)
                content = json_line['content']
                prompt = json_line['prompt']
                # Model Input is aslike: [CLS] Prompt [SEP] Content [SEP]
                # It include three summary tokens.
                if max_seq_len <= len(prompt) + 3:
                    raise ValueError("The value of max_seq_len is too small, please set a larger value")
                max_content_len = max_seq_len - len(prompt) - 3
                if len(content) <= max_content_len:
                    yield json_line
                else:
                    result_list = json_line['result_list']
                    json_lines = []
                    accumulate = 0
                    while True:
                        cur_result_list = []
                        
                        for result in result_list:
                            if result['start'] + 1 <= max_content_len < result['end']:
                                max_content_len = result['start']
                                break
                        
                        cur_content = content[:max_content_len]
                        res_content = content[max_content_len:]
                        
                        while True:
                            if len(result_list) == 0:
                                break
                            elif result_list[0]['end'] <= max_content_len:
                                if result_list[0]['end'] > 0:
                                    cur_result = result_list.pop(0)
                                    cur_result_list.append(cur_result)
                                else:
                                    cur_result_list = [result for result in result_list]
                                    break
                            else:
                                break
                        
                        json_line = {'content': cur_content, 'result_list': cur_result_list, 'prompt': prompt}
                        json_lines.append(json_line)
                        
                        for result in result_list:
                            if result['end'] <= 0:
                                break
                            result['start'] -= max_content_len
                            result['end'] -= max_content_len
                        accumulate += max_content_len
                        max_content_len = max_seq_len - len(prompt) - 3
                        if len(res_content) == 0:
                            break
                        elif len(res_content) < max_content_len:
                            json_line = {'content': res_content, 'result_list': result_list, 'prompt': prompt}
                            json_lines.append(json_line)
                            break
                        else:
                            content = res_content
                    
                    for json_line in json_lines:
                        yield json_line


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


class UIELoss(nn.Module):
    def forward(self, y_pred, y_true):
        # start_prob, end_prob = y_pred
        # start_ids, end_ids = y_true
        start_logits, end_logits = y_pred
        start_ids, end_ids = y_true
        # loss_start = torch.nn.functional.binary_cross_entropy(start_prob, start_ids)
        # loss_end = torch.nn.functional.binary_cross_entropy(end_prob, end_ids)
        
        loss_start = MultilabelCategoricalCrossentropy()(start_logits, start_ids)
        loss_end = MultilabelCategoricalCrossentropy()(end_logits, end_ids)
        
        # loss_start = QFocalLoss(nn.BCEWithLogitsLoss())(start_logits, start_ids)
        # loss_end = QFocalLoss(nn.BCEWithLogitsLoss())(end_logits, end_ids)
        return loss_start + loss_end


class SpanEvaluator(Callback):
    """SpanEvaluator computes the precision, recall and F1-score for span detection.
    """
    
    def __init__(self, valid_dataloader, valid_len, best_weight_path = None, EMA = None):
        self.num_infer_spans = 0
        self.num_label_spans = 0
        self.num_correct_spans = 0
        self.best_val_f1 = 0
        self.valid_dataloader = valid_dataloader
        self.best_weight_path = best_weight_path
        self.valid_len = valid_len
        self.ema_schedule = EMA
    
    def on_epoch_end(self, steps, epoch, logs = None):
        if self.ema_schedule is not None:
            self.ema_schedule.apply_ema_weights()
        f1, precision, recall = self.evaluate(self.valid_dataloader)
        logs['val_f1'] = f1
        if f1 > self.best_val_f1:
            self.best_val_f1 = f1
            if self.best_weight_path is not None and self.best_weight_path != "":
                uie_model.save_weights(self.best_weight_path)
        if self.ema_schedule is not None:
            self.ema_schedule.restore_raw_weights()
        logger.info(f'[val-entity level] f1: {f1:.5f}, p: {precision:.5f} r: {recall:.5f}\n')
        history['epoch'].append(epoch + 1)
        history['f1'].append(f1)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['loss'].append(logs['loss'])
    
    def evaluate(self, dataloder):
        self.reset()
        pbar = tqdm(ncols = 0, total = self.valid_len)
        for x_true, y_true in dataloder:
            start_prob, end_prob = uie_model.predict(*x_true)
            start_ids, end_ids = y_true
            num_correct, num_infer, num_label = self.compute(start_prob, end_prob, start_ids, end_ids)
            self.update(num_correct, num_infer, num_label)
            pbar.update()
            precision, recall, f1 = self.accumulate()
            pbar.set_description(
                'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
            )
        # pbar.close()
        precision, recall, f1 = self.accumulate()
        return f1, precision, recall
    
    def compute(self, start_probs, end_probs, gold_start_ids, gold_end_ids):
        """Computes the precision, recall and F1-score for span detection.
        """
        start_probs = start_probs.cpu().numpy()
        end_probs = end_probs.cpu().numpy()
        gold_start_ids = gold_start_ids.cpu().numpy()
        gold_end_ids = gold_end_ids.cpu().numpy()
        
        pred_start_ids = get_bool_ids_greater_than(start_probs)
        pred_end_ids = get_bool_ids_greater_than(end_probs)
        gold_start_ids = get_bool_ids_greater_than(gold_start_ids.tolist())
        gold_end_ids = get_bool_ids_greater_than(gold_end_ids.tolist())
        num_correct_spans = 0
        num_infer_spans = 0
        num_label_spans = 0
        for predict_start_ids, predict_end_ids, label_start_ids, label_end_ids in zip(
                pred_start_ids, pred_end_ids, gold_start_ids, gold_end_ids):
            [_correct, _infer, _label] = self.eval_span(predict_start_ids, predict_end_ids, label_start_ids,
                                                        label_end_ids)
            num_correct_spans += _correct
            num_infer_spans += _infer
            num_label_spans += _label
        return num_correct_spans, num_infer_spans, num_label_spans
    
    def update(self, num_correct_spans, num_infer_spans, num_label_spans):
        """
        This function takes (num_infer_spans, num_label_spans, num_correct_spans) as input,
        to accumulate and update the corresponding status of the SpanEvaluator object.
        """
        self.num_infer_spans += num_infer_spans
        self.num_label_spans += num_label_spans
        self.num_correct_spans += num_correct_spans
    
    def eval_span(self, predict_start_ids, predict_end_ids, label_start_ids, label_end_ids):
        """
        evaluate position extraction (start, end)
        return num_correct, num_infer, num_label
        input: [1, 2, 10] [4, 12] [2, 10] [4, 11]
        output: (1, 2, 2)
        """
        pred_set = get_span(predict_start_ids, predict_end_ids)
        label_set = get_span(label_start_ids, label_end_ids)
        num_correct = len(pred_set & label_set)
        num_infer = len(pred_set)
        num_label = len(label_set)
        return (num_correct, num_infer, num_label)
    
    def accumulate(self):
        """
        This function returns the mean precision, recall and f1 score for all accumulated minibatches.
        Returns:
            tuple: Returns tuple (`precision, recall, f1 score`).
        """
        precision = float(self.num_correct_spans / self.num_infer_spans) if self.num_infer_spans else 0.
        recall = float(self.num_correct_spans / self.num_label_spans) if self.num_label_spans else 0.
        f1_score = float(2 * precision * recall / (precision + recall)) if self.num_correct_spans else 0.
        return precision, recall, f1_score
    
    def reset(self):
        """
        Reset function empties the evaluation memory for previous mini-batches.
        """
        self.num_infer_spans = 0
        self.num_label_spans = 0
        self.num_correct_spans = 0


def train():
    if not os.path.exists(args.train_path):
        raise ValueError("Please input the correct path of converted training set.")
    if not os.path.exists(args.dev_path):
        raise ValueError("Please input the correct path of converted dev set.")
    parent_dir_best_weight = os.path.dirname(args.best_weight_path)
    if not os.path.exists(parent_dir_best_weight):
        os.makedirs(parent_dir_best_weight)
    parent_dir_train_history = os.path.dirname(args.train_history_path)
    if not os.path.exists(parent_dir_train_history):
        os.makedirs(parent_dir_train_history)
    parent_dir_log = os.path.dirname(args.log_path)
    if not os.path.exists(parent_dir_log):
        os.makedirs(parent_dir_log)
    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)
    if args.batch_size <= 0:
        raise ValueError("Please input the correct batch size.")
    if args.learning_rate <= 0.:
        raise ValueError("Please input the correct learning rate.")
    if args.max_seq_len <= 0:
        raise ValueError("Please input the correct max sequence length.")
    if args.num_epochs <= 0:
        raise ValueError("Please input the correct epochs number.")
    if args.early_stop_patience <= 0:
        raise ValueError("Please input the correct early stop patience.")
    
    uie_model.to(device)
    
    # 数据准备
    train_ds = IEDataset(args.train_path, tokenizer = tokenizer, max_seq_len = args.max_seq_len, fewshot = None)
    dev_ds = IEDataset(args.dev_path, tokenizer = tokenizer, max_seq_len = args.max_seq_len)
    train_dataloader = DataLoader(train_ds, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)
    if args.debug:
        d = []
        for i in dev_ds.dataset:
            if len(i['result_list']) != 0:
                d.append(i)
        dev_ds.dataset = d
    valid_dataloader = DataLoader(dev_ds, batch_size = args.batch_size, collate_fn = collate_fn)
    
    # optimizer = torch.optim.AdamW(lr = args.learning_rate, params = uie_model.parameters())
    optimizer = torch.optim.Adam(uie_model.parameters(), lr = args.learning_rate)
    ema_schedule = extend_with_exponential_moving_average(uie_model, decay = 0.9)
    num_training_steps = len(
        train_dataloader) * args.num_epochs if args.steps_per_epoch is None else args.steps_per_epoch * args.num_epochs
    warmup_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                       num_warmup_steps = num_training_steps * 0.1,
                                                       num_training_steps = num_training_steps,
                                                       last_epoch = -1)
    
    uie_model.compile(
        loss = UIELoss(),
        optimizer = optimizer,
        # scheduler = ema_schedule,
        scheduler = [ema_schedule, warmup_scheduler],
        # scheduler = warmup_scheduler,
        # adversarial_train = {'name': 'vat'},
    )
    summary(uie_model, input_data = next(iter(train_dataloader))[0])
    
    evaluator = SpanEvaluator(valid_dataloader = valid_dataloader,
                              valid_len = len(dev_ds) // args.batch_size,
                              best_weight_path = args.best_weight_path,
                              EMA = ema_schedule
                              )
    # earlystoping = EarlyStopping(monitor = 'val_f1', patience = args.early_stop_patience, verbose = 1, mode = 'max')
    # training_logger = Logger(args.log_path)
    # tensorboard = Tensorboard(args.tensorboard_dir)
    zero_f1, zero_precision, zero_recall = evaluator.evaluate(valid_dataloader)
    logger.info('zero_shot performance - f1:{0:.4f}, precision:{0:.4f}, recall:{0:.4f}\n'.format(zero_f1, zero_precision,
                                                                                           zero_recall))
    uie_model.save_weights("./checkpoints/lung_uie_untrained.pt")
    # logger.info("model runs on {}".format(str(next(uie_model.parameters()).device)))
    # uie_model.fit(train_dataloader, epochs = args.num_epochs, steps_per_epoch = args.steps_per_epoch,
    #               callbacks = [evaluator, earlystoping, training_logger, tensorboard])
    
    # pd.DataFrame(history, index = None).to_csv(args.train_history_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-b", "--batch_size", default = 16, type = int,
                        help = "The batch size in the train.")
    parser.add_argument("-l", "--learning_rate", default = 1e-5, type = float,
                        help = "The path of data that you wanna save.")
    parser.add_argument("-t", "--train_path", default = "./data/target_data/train.txt", type = str,
                        help = "The converted training set path after step 2.")
    parser.add_argument("-v", "--dev_path", default = "./data/target_data/dev.txt", type = str,
                        help = "The converted dev set path after step 2.")
    parser.add_argument("-m", "--max_seq_len", default = 256, type = int,
                        help = "The max sequence length in training.")
    parser.add_argument("-e", "--num_epochs", default = 256, type = int,
                        help = "The max epoch in training if no early stopping strategy applied.")
    parser.add_argument("-p", "--early_stop_patience", type = int, default = 5,
                        help = "The early stop patience if monitored value not improved.")
    parser.add_argument("--log_path", default = "./log/train.log",
                        type = str, help = "The path that you wanna save training log.")
    parser.add_argument("--tensorboard_dir", default = "./tensorboard/",
                        type = str, help = "The directory that you wanna save tensorboard log.")
    parser.add_argument("--best_weight_path", default = "./best_weight.pt",
                        type = str, help = "The name that you wanna save best training weight.")
    parser.add_argument("--train_history_path", default = "./train_log.csv",
                        type = str, help = "The path that you wanna save training f1 csv.")
    parser.add_argument("--debug", action = 'store_true',
                        help = "Precision, recall and F1 score are calculated without negative samples.")
    parser.add_argument("--steps_per_epoch", type = int, default = None, help = "The steps per epoch.")
    
    args = parser.parse_args()
    
    train()
