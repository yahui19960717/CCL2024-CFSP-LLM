
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

def get_task23(file_pre, test_B):
    # file_list = [file_pre+str(i)+".json" for i in range(1,12)]
    file_list = [file_pre+".json"]
    # import pdb;pdb.set_trace()
    task2_results, task3_results, data_dic = [], [], {}
    kong, error = 0, 0
    # 获取句子id
    for i in test_B:
        id_sen = i['sentence_id']
        if id_sen not in data_dic.keys():
            data_dic[id_sen] = i
    for file in file_list:
        temp_results = read(file)
        # 根据文本获取span的下标
        for ele in temp_results:
            id_sen = ele[0]
            try:
                spans = eval(ele[1])
            except:
                error += 1
                pass
                
            if len(spans) !=0:
                for span in spans:
                    temp_sen = data_dic[id_sen]
                    text = temp_sen['text']
                    start, end = obtain_index(text, span[0])
                    task2_results.append([id_sen, start, end])
                    task3_results.append([id_sen, start, end, span[1]])
            else:
                kong += 1
    write(task2_results, "./submit/B_task2_test.json")
    # write(task3_results, "B_task3_test.json")
    print(kong, error)

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
    file = "./llm-out-gemini/B_task2_test_org"
    test_B = read_org("org_data/cfn-test-B_org.json")
    get_task23(file, test_B)