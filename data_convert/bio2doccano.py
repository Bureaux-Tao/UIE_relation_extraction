# 数据转换1
import os
import re
import json

en2ch = {'有无胰腺脓肿': '脓肿情况', '有无胰腺坏死': '坏死情况', '有无假性囊肿出血': '假性囊肿出血情况',
         '胰腺形态': '胰腺形态', '血栓形成情况': '血栓形成情况',
         '扩张情况': '扩张情况', '有无胰腺假性囊肿': '假性囊肿情况', '有无胰周渗出积液': '胰周渗出积液情况',
         '积液情况': '胸水与腹水情况', '胰腺胰周炎性改变': '炎性改变情况',
         '有无胰管结石': '胰管结石情况'}


def preprocess(input_path_list, save_path, save_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_path = os.path.join(save_path)
    result = []
    tmp = {}
    tmp['id'] = 0
    tmp['text'] = ''
    tmp['relations'] = []
    tmp['entities'] = []
    # =======先找出句子和句子中的所有实体和类型=======
    lines = []
    for file in input_path_list:
        with open(file, 'r', encoding = 'utf-8') as fp:
            lines.extend(list(fp.readlines()))
        del fp
    
    texts = []
    entities = []
    words = []
    entity_tmp = []
    entities_tmp = []
    entity_label = ''
    for line in lines:
        line = line.strip().split("\t")
        if len(line) == 2:
            word = line[0]
            label = line[1]
            words.append(word)
            
            if "B-" in label:
                entity_tmp.append(word)
                entity_label = en2ch[label.split("-")[-1]]
            elif "I-" in label:
                entity_tmp.append(word)
            if (label == 'O') and entity_tmp:
                if ("".join(entity_tmp), entity_label) not in entities_tmp:
                    entities_tmp.append(("".join(entity_tmp), entity_label))
                entity_tmp, entity_label = [], ''
        else:
            if entity_tmp and (("".join(entity_tmp), entity_label) not in entities_tmp):
                entities_tmp.append(("".join(entity_tmp), entity_label))
                entity_tmp, entity_label = [], ''
            
            texts.append("".join(words))
            entities.append(entities_tmp)
            words = []
            entities_tmp = []
    
    # ==========================================
    # =======找出句子中实体的位置=======
    i = 0
    for text, entity in zip(texts, entities):
        
        if entity:
            ltmp = []
            for ent, type in entity:
                for span in re.finditer(ent, text):
                    start = span.start()
                    end = span.end()
                    ltmp.append((type, start, end, ent))
                    # print(ltmp)
            ltmp = sorted(ltmp, key = lambda x: (x[1], x[2]))
            for j in range(len(ltmp)):
                # tmp['entities'].append(["".format(str(j)), ltmp[j][0], ltmp[j][1], ltmp[j][2], ltmp[j][3]])
                tmp['entities'].append(
                    {"id": j, "start_offset": ltmp[j][1], "end_offset": ltmp[j][2], "label": ltmp[j][0]})
        else:
            tmp['entities'] = []
        tmp['id'] = i
        tmp['text'] = text
        result.append(tmp)
        tmp = {}
        tmp['id'] = 0
        tmp['text'] = ''
        tmp['relations'] = []
        tmp['entities'] = []
        i += 1
    
    with open(data_path + save_name, 'w', encoding = 'utf-8') as fp:
        fp.write("\n".join([json.dumps(i, ensure_ascii = False) for i in result]))


preprocess(["./data/ori_data/changhai/labeled_ct_mr_train.txt",
            "./data/ori_data//changhai/labeled_ct_mr_test.txt"],
           "../data/doccano_data/",
           "changhai.jsonl")
