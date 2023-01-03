import argparse
import json
import os
import time

from utils import logger


def do_convert():
    args = parser.parse_args()
    if not os.path.exists(args.colabeler_file):
        raise ValueError("Please input the correct path of colabeler file.")
    parent_dir_doccano = os.path.dirname(args.doccano_file)
    if not os.path.exists(parent_dir_doccano):
        os.makedirs(parent_dir_doccano)
    
    tic_time = time.time()
    with open(args.colabeler_file, 'r', encoding = 'utf8') as f1:
        data = json.load(f1)
        big_text = data["content"]
        relations = sorted(list(filter(lambda x: x is not None, data["outputs"]["annotation"]["R"]))[1:],
                           key = lambda item: (int(item["from"]), int(item["to"])))
        # {'name': '部位与器官', 'from': 23, 'to': 21, 'arg1': 'Arg1', 'arg2': 'Arg2'}
        entities = sorted(list(filter(lambda x: x is not None, data["outputs"]["annotation"]["T"]))[1:],
                          key = lambda item: (int(item["start"]), int(item["end"]), int(item["id"])))
        # {'type': 'T', 'name': '部位与器官', 'value': '右肺下叶外基底段', 'start': 20, 'end': 28, 'attributes': [], 'id': 21}
    
    final = []
    big_text = big_text.replace("\r", "")
    big_text = ("\n" + big_text + "\n").replace("(", "（").replace(")", "）").replace("*", "×").replace(
        "+", "＋")
    count = 0
    offset = 0
    while big_text != "":
        text = big_text[0:len(big_text.split("\n")[0])]
        # print(count, text)
        count_entities = 0
        entities_per_text = []
        ids = []
        # print("entities:")
        for entity in entities:
            if entity['end'] <= len(text) + 1 + offset:
                entities_per_text.append(
                    {'id': entity['id'],
                     'label': entity['name'],
                     'start_offset': entity['start'] - offset + 1,
                     'end_offset': entity['end'] - offset + 1})
                ids.append(entity['id'])
                # print(entities_per_text[-1],
                #       text[entities_per_text[-1]['start_offset']:entities_per_text[-1]['end_offset']], entity['start'],
                #       entity['end'], offset)
                count_entities += 1
            else:
                break
        entities = entities[count_entities:]
        ids = list(set(ids))
        relations_per_text = []
        # print("relations:")
        for relation in relations:
            if relation['from'] in ids or relation['to'] in ids:
                relations_per_text.append({
                    'from_id': relation['from'],
                    'to_id': relation['to'],
                    'type': relation['name']
                })
                # print(relations_per_text[-1])
        
        offset += len(text) + 1
        big_text = big_text[len(text) + 1:]
        
        count += 1
        # print("****************************************")
        if count % 100 == 0:
            logger.info("Handled {} examples.".format(count))
        final.append({
            "text": text,
            "entities": entities_per_text,
            "relations": relations_per_text
        })
    
    if os.path.exists(args.doccano_file):
        os.remove(args.doccano_file)
    with open(args.doccano_file, "a", encoding = "utf-8") as f:
        for i in final:
            if i['text'] != "":
                f.write(json.dumps(i, ensure_ascii = False) + '\n')
        # break
    logger.info('Finished! It takes %.2f seconds' % (time.time() - tic_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--colabeler_file", default = "../data/export.json", type = str,
                        help = "The colabeler file exported from doccano platform.")
    parser.add_argument("--doccano_file", default = "../data/doccano.jsonl", type = str,
                        help = "The doccano file exported from doccano platform.")
    args = parser.parse_args()
    do_convert()
