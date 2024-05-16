# 环境依赖
 * 需要先安装依赖环境：gemini的安装环境，主要是google.generativeai这个包
```bash
pip install "python>=3.9.0"
pip install -r requirements.txt 
```
 * 注意⚠️!!先修改`llm_task1_gemini.py、llm_task2_gemini.py和llm_task3_gemini.py`文件中的api_key设置

# 数据准备
将相关数据已放于`data`和`org_data`目录下

# 在LLM文件夹下运行
 * 框架识别任务：
```bash
python llm_task1_gemini.py
```
 * 论元范围识别任务：
```bash
先使用LLM运行：
python llm_task2_gemini.py
运行完成后进行后处理：
python ./llm-out-gemini/deal_task2.py
```
 * 论元角色识别任务：
```bash
先使用LLM运行：
python llm_task3_gemini.py
运行完成后进行后处理：
python ./llm-out-gemini/deal_task3.py
```
* 运行完成在./submit文件夹下可以看到生成的三个文件`B_task1_test.json, B_task2_test.json, B_task3_test.json`

* 获得submit包
```bash
cd ./submit
zip submit.zip B_task1_test.json B_task2_test.json B_task3_test.json
```
