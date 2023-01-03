import json


def search(id, entities, text):
    for i in entities:
        if id == i['id']:
            return text[i["start_offset"]: i["end_offset"]]


d = json.loads('{"id":33731,"text":"肝癌术后复发微创治疗后？术区积液。 肝硬化，门静脉、脾静脉增宽。脾脏肿大。 胰腺体尾部多发斑点钙化。 左肾小囊肿。 两下肺结节影。","entities":[{"id":100574,"label":"疾病与症状","start_offset":14,"end_offset":16},{"id":100575,"label":"部位与器官","start_offset":18,"end_offset":19},{"id":100576,"label":"疾病与症状","start_offset":19,"end_offset":21},{"id":100577,"label":"部位与器官","start_offset":22,"end_offset":25},{"id":100578,"label":"部位与器官","start_offset":26,"end_offset":29},{"id":100579,"label":"疾病与症状","start_offset":29,"end_offset":31},{"id":100580,"label":"部位与器官","start_offset":32,"end_offset":34},{"id":100581,"label":"疾病与症状","start_offset":34,"end_offset":36},{"id":100582,"label":"部位与器官","start_offset":38,"end_offset":43},{"id":100583,"label":"数量","start_offset":43,"end_offset":45},{"id":100584,"label":"疾病与症状","start_offset":45,"end_offset":49},{"id":100585,"label":"部位与器官","start_offset":50,"end_offset":53},{"id":100586,"label":"疾病与症状","start_offset":53,"end_offset":56},{"id":100587,"label":"部位与器官","start_offset":57,"end_offset":61},{"id":100588,"label":"疾病与症状","start_offset":61,"end_offset":64}],"relations":[{"id":12774,"from_id":100579,"to_id":100577,"type":"部位与器官"},{"id":12775,"from_id":100579,"to_id":100578,"type":"部位与器官"},{"id":12776,"from_id":100581,"to_id":100580,"type":"部位与器官"},{"id":12777,"from_id":100584,"to_id":100582,"type":"部位与器官"},{"id":12778,"from_id":100584,"to_id":100583,"type":"数量"},{"id":12779,"from_id":100586,"to_id":100585,"type":"部位与器官"},{"id":12780,"from_id":100588,"to_id":100587,"type":"部位与器官"},{"id":12781,"from_id":100574,"to_id":100573,"type":"部位与器官"},{"id":12782,"from_id":100576,"to_id":100575,"type":"部位与器官"}]}')
print(d['text'])
for i in d['entities']:
    print(d['text'][i["start_offset"]:i["end_offset"]])
for i in d['relations']:
    print(search(i['from_id'], d['entities'], d['text']), "的", i["type"], "是",
          search(i['to_id'], d['entities'], d['text']))