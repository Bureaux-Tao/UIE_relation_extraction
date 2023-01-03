import argparse
import json
import os
import re
from tqdm import tqdm


def search_id(id, entities):
    return list(filter(lambda item: item['id'] == id, entities))[0]


def split_and_get_index(text):
    l = []
    text = text.replace("\r", "")
    to_be_find = ("\n" + text + "\n").replace("(", "（").replace(")", "）").replace("*", "×").replace(
        "+", "＋")
    split_sentenses = list(set(to_be_find.split('\n')))
    
    for i, item in enumerate(split_sentenses):
        starts = [each.start() for each in re.finditer("\n" + item + "\n", to_be_find)]  # [0, 8]
        ends = [start + len(item) for start in starts]
        span = [(start, end) for start, end in zip(starts, ends)]
        # print(i + 1, item)
        # print(span)
        for start, end in span:
            l.append((text[start:end], (start, end)))
    return sorted(l, key = lambda item: item[1][0])


def is_in_entities(s, e, entities):
    for i in entities:
        if s == i["start_offset"] and e == i["end_offset"]:
            return True
    return False


def find_id(s, e, entities):
    for i in entities:
        if s == i["start_offset"] and e == i["end_offset"]:
            return i["id"]


def do_convert():
    args = parser.parse_args()
    if not os.path.exists(args.spo_file):
        raise ValueError("Please input the correct path of spo file.")
    parent_dir_doccano = os.path.dirname(args.spo_file)
    if not os.path.exists(parent_dir_doccano):
        os.makedirs(parent_dir_doccano)
    with open(args.spo_file, 'r', encoding = 'utf8') as f1:
        final = f1.readlines()
        final = [json.loads(i.strip()) for i in final]
    
    count = 1
    
    if os.path.exists(args.doccano_file):
        os.remove(args.doccano_file)
    with open(args.doccano_file, "a", encoding = "utf-8") as f:
        for d in final:
            entities = []
            relations = []
            for spo in d['spo_list']:
                object_start_offset = d['text'].find(spo['object'])
                object_end_offset = d['text'].find(spo['object']) + len(spo['object'])
                if not is_in_entities(object_start_offset, object_end_offset, entities):
                    entities.append(
                        {"id": count, "start_offset": object_start_offset, "end_offset": object_end_offset,
                         "label": spo['object_type']})
                    object_id = count
                    count += 1
                subject_start_offset = d['text'].find(spo['subject'])
                subject_end_offset = d['text'].find(spo['subject']) + len(spo['subject'])
                if not is_in_entities(subject_start_offset, subject_end_offset, entities):
                    entities.append(
                        {"id": count, "start_offset": subject_start_offset, "end_offset": subject_end_offset,
                         "label": spo['subject_type']})
                    subject_id = count
                    count += 1
                else:
                    subject_id = find_id(subject_start_offset, subject_end_offset, entities)
                relations.append({"id": count, "from_id": subject_id, "to_id": object_id, "type": spo["predicate"]})
                count += 1
            
            r = {"text": d["text"], "entities": entities, "relations": relations}
            # print(json.dumps(r, ensure_ascii = False))
            f.write(json.dumps(r, ensure_ascii = False) + '\n')
        # break
    
    # start = 66921
    # end = 66929
    # print(big_text[start:end])
    # print(big_text.replace("\n", "")[start:end])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--spo_file", default = "../data/export.json", type = str,
                        help = "The colabeler file exported from doccano platform.")
    parser.add_argument("--doccano_file", default = "../data/doccano.jsonl", type = str,
                        help = "The doccano file exported from doccano platform.")
    args = parser.parse_args()
    do_convert()
