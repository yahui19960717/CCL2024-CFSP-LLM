import google.generativeai as genai
import os
import json
import sys
from tqdm import tqdm
import random

API_KEY = 'xx' # 自行配置 
model = 'gemini-pro'

genai.configure(api_key=API_KEY, transport='rest')

# Set up the model
generation_config = {
  "temperature": 0,
  "top_p": 0.95,
  "top_k": 0,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  },
]

client = genai.GenerativeModel(model_name="gemini-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)



def read_org(file):
    with open(file, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return data

def read(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
def obtain_gold_spans_dic(args):
    dic = {}
    for i in args:
        if i[0] not in dic.keys():
            dic[i[0]] = []
            dic[i[0]].append([i[1],i[2]])
        else:
            dic[i[0]].append([i[1], i[2]])
    return dic

def get_frames(file):
    dic = {}
    fs = read(file)
    for f in fs:
        fes = f['fes']
        if f['frame_name'] not in dic.keys(): 
            dic[f['frame_name']] = []
            for fe in fes:
                dic[f['frame_name']].append(fe['fe_name'])
    print(len(dic))
    return dic

def obtain_gold_frame_dic(args):
    dic = {}
    for i in args:
        if i[0] not in dic.keys():
            dic[i[0]] = i[1]
    return dic

def get_spans_text_dic(index_span, sen_text):
    text_span = []
    for i in index_span:
        text = sen_text[i[0]:i[1]+1]
        text_span.append(text)
    return text_span

results = []
testA = read_org("org_data/cfn-test-B_org.json")
gold_frame = read("data/gold_frame.json")
gold_frame_dic = obtain_gold_frame_dic(gold_frame)
gold_spans = read("data/gold_spans.json")
gold_spans_dic = obtain_gold_spans_dic(gold_spans)
frame_info = get_frames("data/frame_info.json")
result_path = "./llm-out-gemini/B_task3_test_org.json"
random.seed(42)

if os.path.exists(result_path):
    results = read(result_path)
    final_id = results[-1][0]
    for i in range(len(testA)):
      if testA[i]["sentence_id"] == final_id:
        testA = testA[i+1:]       
        break         
for sen in tqdm(testA):
    id_sen = sen['sentence_id']
    text = sen['text']
    temp_target = [text[i['start']:i['end']+1] for i in sen['target']]
    frame = gold_frame_dic[id_sen]
    try:
        spans = gold_spans_dic[id_sen]
        if len(temp_target) == 1:
            target = temp_target[0]
        elif len(temp_target) == 2:
            target = "...".join([temp_target[0],temp_target[1]])
        spans_text = get_spans_text_dic(spans, text)
        prompt = f'''作为汉语框架语义学家，你的任务是根据给定的句子的目标词、框架以及目标词在句子中的相关论元成分，从候选的论元标签中为它们分配相应的论元标签。请注意，论元成分是指句子中与目标词直接相关的部分，而论元标签描述了这些成分的具体角色。\
        \n输出格式：每个输出元素应该是一个列表，包含两个字符串元素：\n一个是论元成分,一个是论元标签，即[[论元成分1，论元标签1],[论元成分2，论元标签2],...]。\n只需要以Python数组的形式输出，不要添加其他内容，如果没有相应的标签就返回''。\
        \n文本：{"".join(sen['text'])}\n目标词："{"".join(sen['target'][0])}"\
        \n框架名称:{frame}\n论元标签：\n{frame_info[frame]}\n{spans_text}\
        \n示例：\n输入：\n文本：['运动会', '闭幕', '后', '，', '他们', '将', '在', '北京', '继续', '逗留', '两', '天', '，', '同', '中国', '有关', '方面', '开展', '交流', '活动', '并', '参观', '游览', '，', '于', '１３日', '返回', '日本', '。']\
        \n目标词：['返回']\
        \n框架名称:{"到达"}\n论元标签：\n{frame_info['到达']}\n论元成分：\n{['他们','北京','于13日', '日本']}\
        \n输出:\n[['他们', '转移体'], ['北京',''], ['于１３日', '时间'], ['日本', '终点']]'''

        completion = client.generate_content(prompt)
        res = completion.text
        results.append([sen['sentence_id'], res.strip('"')])
        # import pdb;pdb.set_trace()
        json.dump(results, open(result_path, 'w', encoding="utf-8"), indent=0, ensure_ascii=False)  
    except:
        pass

