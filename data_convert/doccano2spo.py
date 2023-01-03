import argparse
import json
import os


def find_entities(text, entities, relation):
    for entity in entities:
        predicate = relation['type']
        if entity['id'] == relation['from_id']:
            subject = text[entity['start_offset']:entity['end_offset']]
            subject_type = entity['label']
        if entity['id'] == relation['to_id']:
            object = text[entity['start_offset']:entity['end_offset']]
            object_type = entity['label']
    return object, object_type, predicate, subject, subject_type


def do_convert():
    args = parser.parse_args()
    if not os.path.exists(args.doccano_file):
        raise ValueError("Please input the correct path of spo file.")
    parent_dir_spo = os.path.dirname(args.spo_file)
    if not os.path.exists(parent_dir_spo):
        os.makedirs(parent_dir_spo)
    with open(args.doccano_file, 'r', encoding = 'utf8') as f1:
        l = f1.readlines()
        l = [json.loads(i.strip()) for i in l]
        final = []
        for i in l:
            spo_list = []
            relations = i["relations"]
            entities = i["entities"]
            for j in relations:
                object, object_type, predicate, subject, subject_type = find_entities(i["text"], entities, j)
                spo_list.append({
                    'object': object,
                    'object_type': object_type,
                    'predicate': predicate,
                    'subject': subject,
                    'subject_type': subject_type
                })
            final.append({"text": i["text"], "spo_list": spo_list})
    
    if os.path.exists(args.spo_file):
        os.remove(args.spo_file)
    with open(args.spo_file, "a", encoding = "utf-8") as f2:
        for i in final:
            f2.write(json.dumps(i, ensure_ascii = False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--doccano_file", default = "../data/doccano.jsonl", type = str,
                        help = "The doccano file exported from doccano platform.")
    parser.add_argument("--spo_file", default = "../data/export.json", type = str,
                        help = "The output file path.")
    args = parser.parse_args()
    do_convert()
