import google.generativeai as genai
import os
import json
import sys
from tqdm import tqdm
import random

API_KEY = 'xx'# 自行配置
model = 'gemini-pro'
'./llm-out-gemini/B_task2_test_org.json'
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


results = []
testA = read_org("org_data/cfn-test-B_org.json")
word_frmaes_no_exit = read('data/testB_word_frames_union.json')
result_path = "./submit/B_task1_test.json"
dic_word_frames = {key:value for element in word_frmaes_no_exit for key, value in element.items()} 
random.seed(777)
if os.path.exists(result_path):
    results = read(result_path)
    final_id = results[-1][0]
    for i in range(len(testA)):
      if testA[i]["sentence_id"] == final_id:
        testA = testA[i+1:]
        break   
for sen in tqdm(testA):
    text = sen['text']
    temp_target = [text[i['start']:i['end']+1] for i in sen['target']]
    if len(temp_target) == 1:
        target = temp_target[0]
    elif len(temp_target) == 2:
        target = "...".join([temp_target[0],temp_target[1]])
    if target in dic_word_frames.keys():
        prompt = f'''给定句子和其目标词,根据目标词在句子中的语义，从框架集合中选择最符合目标词触发的框架。请以字符串的形式输出。\
        \n示例：输入：\
        \ntext: "双方还就各自的责任、权利以及资金的运作程序等达成了协议。"\ntarget:"达成"\n框架集合："['观点一致', '成就']"\
        \n输出：\n"观点一致"\
        \n给定输入：\ntext:"{text}"\ntarget:"{target}"\n框架集合:"{dic_word_frames[target]}"\
        \n请输出其合适的目标词。'''
        completion = client.generate_content(prompt)
        res = completion.text
        results.append([sen['sentence_id'], res.strip('"')])
        # import pdb;pdb.set_trace()
        json.dump(results, open(result_path, 'w', encoding="utf-8"), indent=0, ensure_ascii=False)  
    else:
        print("error!")


