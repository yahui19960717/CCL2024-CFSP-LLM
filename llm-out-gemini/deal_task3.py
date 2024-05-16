import json
def read(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def read_org(file):
    with open(file, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return data

def write(results, file):
    with open(file, 'w', encoding="utf-8") as out:
        json.dump(results, out, indent=0, ensure_ascii=False)  
   
def union_files(file_list, file_out):
    list_results = []
    for file in file_list:
        temp = read(file)
        list_results.extend(temp)
    
    write(list_results, file_out)

def obtain_index(text, span):
    start = text.find(span)
    end = start+len(span)-1
    return start, end

def obtain_gold_spans_dic(args):
    dic = {}
    for i in args:
        if i[0] not in dic.keys():
            dic[i[0]] = []
            dic[i[0]].append([i[1],i[2]])
        else:
            dic[i[0]].append([i[1], i[2]])
    return dic
def find_index(text, span):
    start = text.find(span)
    end = start+len(span)-1
    return [start, end]
if __name__=="__main__":
    # task3:根据小模型得到的任务2来获得任务3的结果
    results = []
    # 原来的文本，得到文本和id
    testB = read_org("org_data/cfn-test-B_org.json")
    test_B_dic = {i['sentence_id']:i['text'] for i in testB}
    # LLM的输出,获得span文本和label
    file = "./llm-out-gemini/B_task3_test_org.json"
    task3_results = read(file)
    # 真实的spans 小模型获取的
    gold_spans = read("data/gold_spans.json")
    gold_spans_dic = obtain_gold_spans_dic(gold_spans)
    # 对齐LLM的span和真实的span
    for i in task3_results: # 遍历每一个句子的预测
        id_sen = i[0]
        text = test_B_dic[id_sen]
        span_labels = eval(i[1])
        try:
            spans = gold_spans_dic[id_sen]
            for j in range(len(span_labels)):
                temp = find_index(text, span_labels[j][0])
                flag = 0
                for true_label in spans:
                    if temp == true_label and span_labels[j][1]!='':
                        flag = 1
                        results.append([id_sen, true_label[0], true_label[1], span_labels[j][1]])
                if flag == 0:
                    print(span_labels[j])
        except:
            pass
        # assert len(span_labels)==len(spans)
    write(results, "./submit/B_task3_test.json")