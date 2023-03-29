import json
from datetime import datetime
import re

def dmy_to_ymd(d):
    return datetime.strptime(d, '%d %b %Y').strftime('%Y-%m-%d')


with open('../README.md', 'r') as f:
    lines = f.readlines()
    # remove empty line
    lines = [line.strip() for line in lines if line.strip()]
    st = lines.index('# Resources')
    end = lines.index('# Papers')
    lines = lines[st:end]
    print(lines)
    
    # find index of line that start with #
    indexs = [i for i, line in enumerate(lines) if line.startswith('#') and not line.startswith('##')]
    db = {"resources": []}

    # split lines by index
    indexs += [len(lines)]
    for i, idx in enumerate(indexs[:-1]):
        field = lines[idx].strip('##').strip()
        print(field)
        content = lines[idx + 1:indexs[i + 1]]

        second_indexs = [i for i, line in enumerate(content) if line.startswith('##')]
        second_indexs += [len(content)]
        for i, idx in enumerate(second_indexs[:-1]):
            task = content[idx].strip('###').strip()
            second_content = content[idx + 1:second_indexs[i + 1]]
            print(task, len(second_content))

            block_len = 4
            if task == 'Tutorial and Jupyter Notebook':
                block_len = 3
                
            for l in range(0, len(second_content), block_len):
                try:
                    item = second_content[l:l + block_len]

                    obj = {}
                    obj['title'] = item[0][2:-4]
                    obj['authors'] = item[1][1:-3]

                    linkstr = item[2].strip()
                    links = re.findall("\[\[([^\]]*)\]\(([^\)]+)\)\]", linkstr)
                    links2 = {}
                    for name, href in links:
                        links2[name] = href

                    obj['links'] = links2

                    if len(item) == 4:
                        obj['date'] = dmy_to_ymd(item[3].strip())
                    obj['field'] = field
                    obj['task'] = task

                    db['resources'].append(obj)
                except Exception as e:
                    print(item)
                    import ipdb; ipdb.set_trace()
                    raise e

    with open('resource.json', 'w') as fout:
        json.dump(db, fout)
