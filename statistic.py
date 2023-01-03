import argparse
import os
from collections import Counter


def bio():
    with open(args.dataset_path, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
        count = 0
        sentense_len = []
        c = set()
        for i in lines:
            count += 1
            if i.strip('\n') == '':
                sentense_len.append(count)
                count = 0
            else:
                char, label = i.strip('\n').split('\t')
                if label != 'O':
                    c.add(label.split("-")[1])
        categories = {}
        for i in c:
            categories[i] = ""
        print("类别:", categories)
        print("句数:", len(sentense_len))
        print("最长单句样本长度:", max(sentense_len))
        freq = dict(Counter(sentense_len))
        count_large = 0
        for length in sentense_len:
            if length > args.maxlen:
                count_large += 1
        print("大于{}数量:".format(args.maxlen), count_large)
        print("被截断比例:", count_large / len(sentense_len))
        print("\n句子长度数量统计(按长度):(句长, 数量)")
        for index, data_dict in enumerate(sorted(freq.items(), key = lambda d: d[0], reverse = True)):
            print(str(index + 1) + ':', data_dict, '\t', end = "")
            if (index + 1) % 10 == 0:
                print()
        print('\n\n句子长度数量统计(按数量):(句长, 数量)')
        for index, data_dict in enumerate(sorted(freq.items(), key = lambda d: d[1], reverse = True)):
            print(str(index + 1) + ':', data_dict, '\t', end = "")
            if (index + 1) % 10 == 0:
                print()


def doccano():
    with open(args.dataset_path, 'r', encoding = 'utf-8') as f:
        lines = [i.strip() for i in f.readlines()]
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default = "../data/doccano.jsonl", type = str,
                        help = "The dataset file exported from doccano platform.")
    parser.add_argument("--maxlen", default = 256, type = int,
                        help = "The max sequence length to truncate.")
    parser.add_argument('--mode', choices = ['bio', 'doccano'], default = "doccano",
                        help = "select the data type [bio, doccano].")
    args = parser.parse_args()
    
    if args.maxlen <= 0:
        raise ValueError("Please input the correct max sequence length.")
    if not os.path.exists(args.dataset_path):
        raise ValueError("Please input the correct path of dataset.")
    
    if args.mode == 'bio':
        bio()
    elif args.mode == 'doccano':
        doccano()
