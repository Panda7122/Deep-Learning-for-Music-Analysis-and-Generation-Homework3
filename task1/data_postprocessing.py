import json
from symusic import Score
def post_processing(token: dict):
    ret = {
        "tokens":[],
        "ids":[],
        "bytes":token['bytes'],
        "events":[],
        "are_ids_encoded":token['are_ids_encoded'],
        "_ticks_bars":token['_ticks_bars'],
        "_ticks_beats":token['_ticks_beats'],
        "_ids_decoded":token['_ids_decoded'],
    }
    if(len(token['tokens']) == len(token['ids']) 
       and len(token['ids']) == len(token['events'])):
        idx = 0
        while idx < len(token['tokens']):
            if token['tokens'][idx].startswith('Bar'):
                if token['events'][idx]['type_'] != 'Bar':
                    continue
                idx += 1
                ret['tokens'].append(token['tokens'][idx])
                ret['ids'].append(token['ids'][idx])
                ret['events'].append(token['events'][idx])
                pass
            elif token['tokens'][idx].startswith('Position'):
                if token['events'][idx]['type_'] != 'Position':
                    idx += 3
                    print("event_type is not Position")
                    continue
                pos = int(token['tokens'][idx].split('_')[-1])
                if idx+2 >= len(token['tokens']):
                    idx += 3
                    continue
                if (token['events'][idx]['value'] != token['events'][idx]['time'] or 
                    token['events'][idx]['value'] != token['events'][idx]['desc']):
                    print("value should=time=desc")
                    idx += 3
                    continue
                if token['events'][idx]['value'] != pos:
                    idx += 3
                    print(f"position value should= {pos}")
                    continue
                nextTok = token['tokens'][idx+1]
                if not nextTok.startswith('Tempo'):
                    idx += 3
                    continue
                tmp = float(nextTok.split('_')[-1])
                if round(token['events'][idx+1]['value'], 2) != round(tmp, 2):
                    idx += 3
                    print(f"tempo value{token['events'][idx+1]['value']} should= {tmp}")
                    continue
                next2Tok = token['tokens'][idx+2]
                if not next2Tok.startswith('Rest'):
                    idx += 3
                    continue
                res = str(next2Tok.split('_')[-1])
                if token['events'][idx+2]['value'] != res:
                    idx += 3
                    print(f"rest value should= {res}")
                    continue
                for delta in range(3):
                    ret['tokens'].append(token['tokens'][idx+delta])
                    ret['ids'].append(token['ids'][idx+delta])
                    ret['events'].append(token['events'][idx+delta])
                idx += 3
                
    print(len(ret['tokens']))
    print(len(ret['ids']))
    print(len(ret['events']))
    pass
with open('task1/runs/txl_music/samples/sample_datas2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
for token in data:
    post_processing(token)