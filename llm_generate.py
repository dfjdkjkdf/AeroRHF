import os
import json
import sys
import time
import torch
import requests
import threading
import subprocess
import re
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from tqdm import tqdm
from textwrap import dedent, indent
import sys
import os

# 模型配置
URL = 'url'

MODEL_CONFIGS = {
    # Ollama模型
    "Qwen3:32B": {
        "type": "ollama",
        "url": "http://0.0.0.0:11434/api/generate", 
        "model_name": "qwen3:latest",
    },
    "Deepseek-32B:latest": {
        "type": "ollama",
        "url": 'http://10.78.61.211:8197/api/generate',
        "model_name": "Deepseek-32B:latest",
    },
}

# 全局变量
stop_loading = False
# 缓存embedding模型资源
_tokenizer = None
_model = None
_device = None
_global_vectors_dict = {}

def spinner():
    """
    动态loading状态提示
    """
    spinner_chars = "|/-\\"
    while not stop_loading:
        for char in spinner_chars:
            sys.stdout.write(f"\r正在等待模型响应 {char} ")
            sys.stdout.flush()
            time.sleep(30)

def llm_generate_ollama(llm_name, prompt):
    """
    通过Ollama API调用本地部署的大语言模型生成代码
    
    Args:
        llm_name: 模型名称
        prompt: 提示词
    
    Returns:
        生成的文本
    """
    global stop_loading, URL
    stop_loading = False
    
    # 获取模型配置
    model_config = MODEL_CONFIGS.get(llm_name, {})
    if not model_config or model_config["type"] != "ollama":
        raise ValueError(f"无效的Ollama模型名称: {llm_name}")
    
    base_url = model_config[URL]
    model_name = model_config["model_name"]

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    try:
        # 启动loading线程
        loading_thread = threading.Thread(target=spinner)
        loading_thread.start()
        
        response = requests.post(base_url, json=payload)
        response.raise_for_status()
        
        # 停止loading
        stop_loading = True
        loading_thread.join()
        
        # 清除loading行
        sys.stdout.write("\r" + " " * 40 + "\r")
        sys.stdout.flush()
        
        return response.json().get("response", "")
    except Exception as e:
        # 停止loading
        stop_loading = True
        loading_thread.join()
        
        # 清除loading行
        sys.stdout.write("\r" + " " * 40 + "\r")
        sys.stdout.flush()
        
        print(f"API调用失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def llm_generate_code(llm_name, prompt):
    """
    统一的代码生成接口，根据模型类型选择合适的调用方法
    
    Args:
        llm_name: 模型名称
        prompt: 提示词
    
    Returns:
        生成的代码文本
    """
    model_config = MODEL_CONFIGS.get(llm_name)
    if not model_config:
        raise ValueError(f"未找到模型配置: {llm_name}")
    
    if llm_name == 'Qwen3:32B':
        prompt += '<think>'
    
    if model_config["type"] == "ollama":
        return llm_generate_ollama(llm_name, prompt)
    else:
        raise ValueError(f"不支持的模型类型: {model_config['type']}")
    

def get_topk(prompt, topk=3, rag_model='bge-large-zh'):
    """
    从基础运算中检索相似度Topk的函数
    
    Args:
        prompt: 原始提示词
        topk: 返回的相似函数数量
    
    Returns:
        Topk的函数列表
    """
    global _tokenizer, _model, _device, _global_vectors_dict
    
    # 延迟加载模型和分词器（仅第一次调用时加载）
    if _tokenizer is None or _model is None:
        if rag_model == 'bge-large-zh':
            COSINE_TOKENIZE_PATH = './Embedding/bge-large-zh'
            COSINE_MODEL_PATH = './Embedding/bge-large-zh'
        elif rag_model == 'bert-base-chinese':
            COSINE_TOKENIZE_PATH = './LLM/bert-base-chinese'
            COSINE_MODEL_PATH = './LLM/bert-base-chinese'

        if rag_model in ['bge-large-zh', 'bert-base-chinese']:
            try:
                _tokenizer = AutoTokenizer.from_pretrained(COSINE_TOKENIZE_PATH)
                _model = AutoModel.from_pretrained(COSINE_MODEL_PATH)
                _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                _model = _model.to(_device)
            except Exception as e:
                print(f"模型加载失败: {e}")
                return []
    
    try:
        # 获取代码数据库
        with open("./code_rag/ref-rag-data.txt", 'r', encoding='utf-8') as f:
            data = [line for line in f]
        
        if rag_model == 'Qwen3-Embedding-0.6B':
            data_ = {
                "queries": [prompt],
                "documents": data
            }
            resp = requests.post("http://localhost:8999/embed_and_score", json=data_)
            scores = resp.json()['scores']
            results = [(d, s) for d, s in zip(data, scores)]

            results.sort(key=lambda x: x[1], reverse=True)
            new_results = [results[i][0] for i in range(topk)]
            return new_results
        
        # 提取gpt值作为参考数据库
        references = []
        for item in data:
            references.append(item)
        
        # 过滤逻辑
        if not references or topk == 0:
            return []
        
        assert topk <= len(references)
        reference_texts = references
        candidate_text = prompt

        # Tokenize candidate text
        candidate_inputs = _tokenizer(candidate_text, padding=True, return_tensors='pt', truncation=True).to(_device)
        
        # Encode candidate text
        with torch.no_grad():
            candidate_outputs = _model(**candidate_inputs)
            candidate_embedding = candidate_outputs.last_hidden_state[:, 0]
            candidate_embedding = torch.nn.functional.normalize(candidate_embedding, p=2, dim=1)

        results = []
        
        for reference_text in tqdm(reference_texts):
            if reference_text not in _global_vectors_dict:
                # Tokenize reference text
                reference_inputs = _tokenizer(reference_text, padding=True, return_tensors='pt', truncation=True).to(_device)
                
                # Encode reference text
                with torch.no_grad():
                    reference_outputs = _model(**reference_inputs)
                    reference_embedding = reference_outputs.last_hidden_state[:, 0]
                    reference_embedding = torch.nn.functional.normalize(reference_embedding, p=2, dim=1)
                    _global_vectors_dict[reference_text] = reference_embedding.cpu()
            else:
                reference_embedding = _global_vectors_dict[reference_text].to(_device)
            # Compute cosine similarity
            similarity = torch.matmul(reference_embedding, candidate_embedding.T).item()
            results.append((reference_text, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        new_results = [results[i][0] for i in range(topk)]
        return new_results

    except Exception as e:
        print(f"检索相关函数失败: {e}")
        return []

def gcc_compile(code, postfix):
    """编译代码返回结果"""
    t = time.time()
    code_file = f"tmp/temp_gcc_{t}_{postfix}.c"
    exe_file = f"tmp/temp_gcc_{t}_{postfix}"
    
    with open(code_file, 'w', encoding='utf-8') as f:
        f.write(code)
    include_path = 'humaneval_x/includes'
    compile_result = subprocess.run(
        ['gcc', code_file, '-I', include_path, '-L', include_path, '-lstdutils7_V2_01', '-o', exe_file, '-lm', '-w'],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    
    return {
        "success": compile_result.returncode == 0,
        "error": compile_result.stderr.decode('utf-8'),
        "code_file": code_file,
        "exe_file": exe_file
    }

def llm_complie(llm_name, req, prefix_code, generated_code, rag, ITERATION=3):
    # 输入代码,gcc编译，如果编译失败，则大模型修改生成代码
    # 输出修改后的生成代码

    def build_prompt(gcc_error):
        task_description = dedent('''
            假设你是一位航天软件工程师，请根据gcc编译信息修改代码，要求如下：
                1. 根据 #编译error信息 对 #生成代码 进行完善;
                2. 不要修改 #前置代码，仅修改 #生成代码，注意括号;
                3. 修改后的代码仍需与 #需求 一致;
                4. 将修改后的生成代码放到 ```c```。''')
        prompt = dedent(f'''
            {indent_except_first(task_description, '            ')}
            相关信息如下：
            #编译error信息: 
                {indent_except_first(gcc_error, '                ')}
            #前置代码: 
                {indent_except_first(prefix_code, '                ')}
            #生成代码：
                {indent_except_first(generated_code, '                ')}
            #需求：
                {indent_except_first(req, '                ')}
            #检索函数：
                {indent_except_first(rag, '                ')}
            修改后的 #生成代码 是：
        ''')
        return prompt

    rag = '\n'.join(rag)
    original_generated_code = generated_code
    iteration_history = []
    for iteration in range(ITERATION):
        try:
            # 编译完整代码
            code = prefix_code + generated_code
            compile_result = gcc_compile(code, iteration) 
            iteration_history.append({
                "iteration": iteration + 1,
                "code": code,
                "compile_error": compile_result["error"]
            })

            # 如果成功，返回生成代码
            if compile_result["success"]:
                print(f'迭代{iteration+1}/{ITERATION}，编译成功！')
                return generated_code
            # 如果失败，大模型修改代码，并提取生成代码
            else:
                gcc_error = compile_result["error"]
                print(f'迭代{iteration+1}/{ITERATION}，编译错误：\n{gcc_error}')
                prompt = build_prompt(gcc_error)
                response = llm_generate_code(llm_name, prompt)
                print(response)
                generated_code = extract_generated_code(response)
        except Exception as e:
            print(f"处理异常：{str(e)}")
            import traceback
            traceback.print_exc()
            return original_generated_code
    return original_generated_code

# v1: 基于checker
# def static_check(code, postfix):
#     """编译代码返回结果"""
#     code_file = f"tmp/temp_static_{postfix}.c"
    
#     with open(code_file, 'w', encoding='utf-8') as f:
#         f.write(code)
#     results = check_main(code_file) # 参考checker
#     error_msg = ""
#     count = 0
#     for key in results:
#         if isinstance(results[key], list) and len(results[key]) > 0:
#             error_msg += f'{key}:\n'
#             for error in results[key]:
#                 error_msg += f'    {str(error)}\n'
#                 count += 1
#     return {'error': error_msg, 'count': count}

# v2：基于cppchecker
def static_check(code, postfix, count_range=None):
    """编译代码返回结果"""
    code_file = f"tmp/temp_static_{postfix}.c"
    
    with open(code_file, 'w', encoding='utf-8') as f:
        f.write(code)
    static_result = subprocess.run(
        ['cppcheck', '--addon=cppchecker/misra.json', code_file],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    count = 0
    # error = ''
    for msg in static_result.stderr.decode('utf-8').split('\n'):
        if code_file in msg:
            if count_range:
                s, e = count_range
                line_num = re.search(r'\.c:(.*?):', msg).group(1)
                try:
                    if int(line_num) >= s and int(line_num) < e:
                        count += 1
                        # error += msg + '\n'
                except:
                    print(f'It is not line num: {line_num}')
            else:
                count += 1
    # return {'error': error, 'count': count}

    return {'error': static_result.stderr.decode('utf-8'), 'count': count}

def llm_static(llm_name, req, prefix_code, generated_code, rag, ITERATION=3, TOLERANCE=3, w_compile=False, w_selfcorrect=True):
    # 输入代码, 静态分析，如果有格式错误，则大模型修改生成代码
    # 输出修改后的生成代码
    def build_prompt(check_error):
        if not w_selfcorrect:
            task_description = dedent('''
                假设你是一位航天软件工程师，请根据静态分析信息修改代码，要求如下：
                    1. 根据 #静态分析error信息 对 #生成代码 进行完善;
                    2. 如果有gcc error错误，根据 #编译error信息 对 #生成代码 进行完善;
                    2. 针对 #前置代码 #静态分析error信息 可以忽略；
                    3. 不要修改 #前置代码，仅修改 #生成代码，注意括号;
                    4. 修改后的代码仍需与 #需求 一致;
                    5. 将修改后的生成代码放到 ```c```。
            ''')
        else:
            task_description = dedent('''
                假设你是一位航天软件工程师，请根据静态分析信息修改代码，要求如下：
                    1. 根据 #静态分析error信息 对 #生成代码 进行完善;
                    2. 根据 #需求 和 #生成代码 的一致性对 #生成代码 进行完善;
                    3. 如果有gcc error错误，根据 #编译error信息 对 #生成代码 进行完善;
                    4. 针对 #前置代码 的 #静态分析error信息 可以忽略；
                    5. 不要修改 #前置代码，仅修改 #生成代码，注意括号;
                    6. 修改后的代码仍需与 #需求 一致;
                    7. 要求控制流和数据流尽可能与需求一一对应，如果你觉得一致性非常好，则无需修改;
                    8. 将修改后的生成代码放到 ```c```。
            ''')
        prompt = dedent(f'''
            {indent_except_first(task_description, '            ')}
            相关信息如下：
            #静态分析error信息: 
                {indent_except_first(check_error, '                ')}
            #编译error信息: 
                {indent_except_first(gcc_error, '                ')}
            #前置代码: 
                {indent_except_first(prefix_code, '                ')}
            #生成代码：
                {indent_except_first(generated_code, '                ')}
            #需求：
                {indent_except_first(req, '                ')}
            #检索函数：
                {indent_except_first(rag, '                ')}
            修改后的 #生成代码 是：
        ''')
        return prompt

    rag = '\n'.join(rag)
    original_generated_code = generated_code
    iteration_history = []
    for iteration in range(ITERATION):
        try:
            # 更新完整代码
            code = prefix_code + '\n' + generated_code
            # code = generated_code
            start = len(prefix_code.splitlines(keepends=True)) + 1
            end = len(prefix_code.splitlines(keepends=True)) + len(generated_code.splitlines(keepends=True)) + 2
            print(f'静态分析范围：{start}，{end}')
            check_result = static_check(code, iteration, count_range=[start, end]) 
            if w_compile:
                compile_result = gcc_compile(code, iteration)

                if compile_result["success"]:
                    gcc_error = ''
                else:
                    gcc_error = compile_result["error"]
            else:
                gcc_error = ''

            iteration_history.append({
                "iteration": iteration + 1,
                "code": code,
                "check_error": check_result['error']
            })

            # 如果成功，返回生成代码
            if check_result["count"] <= TOLERANCE and gcc_error == '':
                print(f'迭代{iteration+1}/{ITERATION}，静态分析成功！')
                return generated_code
            # 如果失败，大模型修改代码，并提取生成代码
            else:
                check_error = check_result["error"]
                print(f'迭代{iteration+1}/{ITERATION}，静态分析错误：\n{check_error}')
                if w_compile:
                    print(f'迭代{iteration+1}/{ITERATION}，编译错误：\n{gcc_error}')
                prompt = build_prompt(check_error)
                response = llm_generate_code(llm_name, prompt)
                print(response)
                generated_code = extract_generated_code(response)
        except Exception as e:
            print(f"处理异常：{str(e)}")
            import traceback
            traceback.print_exc()
            return original_generated_code
    return original_generated_code


def llm_selfcorrect(llm_name, req, prefix_code, generated_code, rag, ITERATION=3, w_compile=False):
    # 输入代码,自我校对，如果一致性较差，则大模型修改生成代码
    # 输出修改后的生成代码

    def build_prompt():
        if not w_compile:
            task_description = dedent('''
                假设你是一位航天软件工程师，请评价并优化代码，要求如下：
                    1. 根据 #需求 和 #生成代码 的一致性对 #生成代码 进行完善;
                    2. 不要修改 #前置代码，仅修改 #生成代码，注意括号;
                    3. 要求控制流和数据流尽可能与需求一一对应，如果你觉得一致性非常好，则无需修改;
                    4. 将修改后的生成代码放到 ```c```，如无需修改，则```c```为空。''')
            prompt = dedent(f'''
                {indent_except_first(task_description, '            ')}
                相关信息如下：
                #前置代码: 
                    {indent_except_first(prefix_code, '                ')}
                #生成代码：
                    {indent_except_first(generated_code, '                ')}
                #需求：
                    {indent_except_first(req, '                ')}
                #检索函数：
                    {indent_except_first(rag, '                ')}
                修改后的 #生成代码 是：
            ''')
        else:
            compile_result = gcc_compile(code, iteration)
            if compile_result["success"]:
                gcc_error = ''
            else:
                gcc_error = compile_result["error"]
            task_description = dedent('''
                假设你是一位航天软件工程师，请评价并优化代码，要求如下：
                    1. 根据 #需求 和 #生成代码 的一致性对 #生成代码 进行完善;
                    2. 根据 #编译error信息 对 #生成代码 进行完善;
                    2. 不要修改 #前置代码，仅修改 #生成代码，注意括号;
                    3. 要求控制流和数据流尽可能与需求一一对应，如果你觉得一致性非常好，则无需修改;
                    4. 将修改后的生成代码放到 ```c```，如无需修改，则```c```为空。''')
            prompt = dedent(f'''
                {indent_except_first(task_description, '            ')}
                相关信息如下：
                #前置代码: 
                    {indent_except_first(prefix_code, '                ')}
                #生成代码：
                    {indent_except_first(generated_code, '                ')}
                #需求：
                    {indent_except_first(req, '                ')}
                #检索函数：
                    {indent_except_first(rag, '                ')}
                #编译error信息: 
                    {indent_except_first(gcc_error, '                ')}
                修改后的 #生成代码 是：
            ''')
        return prompt

    rag = '\n'.join(rag)
    original_generated_code = generated_code
    iteration_history = []
    for iteration in range(ITERATION):
        try:
            # 完整代码
            code = prefix_code + generated_code

            prompt = build_prompt()
            response = llm_generate_code(llm_name, prompt)
            print(response)
            new_generated_code = extract_generated_code(response)

            if new_generated_code.strip() == generated_code.strip() or new_generated_code.strip() == '':
                return generated_code
                print(f'迭代{iteration+1}/{ITERATION}，更新成功！')
            else:
                generated_code = new_generated_code
                print(f'迭代{iteration+1}/{ITERATION}，继续更新！')
        except Exception as e:
            print(f"处理异常：{str(e)}")
            import traceback
            traceback.print_exc()
            return original_generated_code
    return generated_code


def build_rag_prompt(req):
    task_description = dedent(f'''
        假设你是一位航天软件工程师，正在根据需求写代码，需要从库函数中检索相关函数。
    ''')
    prompt = dedent(f'''
        {indent_except_first(task_description, '        ')}
        需求如下：
            {indent_except_first(req, '            ')}
        请问下面函数是否能用于代码中？
    ''')
    return prompt

def build_codegenerated_prompt(req, rag, prefix_code):
    task_description = dedent('''
        假设你是一位航天软件工程师，请根据需求生成代码，要求如下：
            1. 代码需与 #需求 一致;
            2. 尽可能使用 #检索函数 生成代码；
            3. 仅生成 #前置代码 后的代码，注意括号;
            4. 如果 #前置代码 为空，则生成完整的代码;
            5. 将生成代码放到 ```c```。''')
    rag = '\n'.join(rag)
    prompt = dedent(f'''
        {indent_except_first(task_description, '        ')}
        相关信息如下：
        #需求：
            {indent_except_first(req, '            ')}
        #检索函数：
            {indent_except_first(rag, '            ')}
        #前置代码：
            {indent_except_first(prefix_code, '            ') }
        生成代码是：
    ''')
    return prompt

def extract_generated_code(code):
    # 尝试从代码块中提取C代码
    code_pattern = r"```c?\s*([\s\S]*?)\s*```"
    matches = re.findall(code_pattern, code)
    if matches:
        code = matches[0]
    
    # 过滤空行
    return '\n'.join([line for line in code.split('\n') if line.strip()])

def indent_except_first(text, prefix):
    return ''.join([l if i == 0 else prefix + l for i, l in enumerate(text.splitlines(True))])

if __name__=="__main__":
    import argparse
    
    # 切换到脚本所在目录，确保相对路径正确
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"工作目录已切换到: {script_dir}")
    
    parser = argparse.ArgumentParser(description='代码生成智能体系统')
    parser.add_argument('--input', type=str, default='code_ref/ref-test.jsonl', help='输入数据文件路径')
    parser.add_argument('--output', type=str, default='code_gen/gen-test.jsonl', help='输出数据文件路径')
    parser.add_argument('--model', type=str, default='Deepseek-32B:latest', help='使用的模型名称')
    parser.add_argument('--url', type=str, default='url', help='使用的url')
    parser.add_argument('--rag_topk', type=int, default=3, help='检索相似函数的数量')
    parser.add_argument('--rag_model', type=str, default='bge-large-zh', choices=['bge-large-zh', 'bert-base-chinese', 'Qwen3-Embedding-0.6B'], help='检索模型')
    parser.add_argument('--llm_compile_iteration', type=int, default=3, help='编译迭代轮数')
    parser.add_argument('--llm_static_iteration', type=int, default=3, help='静态分析迭代轮数')
    parser.add_argument('--llm_static_tolerance', type=int, default=0, help='静态分析容忍错误次数')
    parser.add_argument('--llm_selfcorrect_iteration', type=int, default=3, help='自我校对迭代轮数')
    # 添加可选步骤的控制参数
    parser.add_argument('--use_rag', action='store_true', help='是否使用RAG检索')
    parser.add_argument('--use_compile', action='store_true', help='是否进行编译和错误修复')
    parser.add_argument('--use_static', action='store_true', help='是否进行静态检查')
    parser.add_argument('--use_selfcorrect', action='store_true', help='是否进行自我校对')
    parser.add_argument('--use_static_w_compile', action='store_true', help='是否在静态检查中添加编译分析')
    parser.add_argument('--use_static_w_selfcorrect', action='store_true', help='是否在静态检查中添加自我矫正')
    parser.add_argument('--use_selfcorrect_w_compile', action='store_true', help='是否在自我校对中添加编译分析')
    parser.add_argument('--continue_generate', action='store_true', help='是否继续生成')
    args = parser.parse_args()
    print(args)

    URL = args.url

    try:
        # 读取数据
        datas = []
        with open(args.input, 'r') as f:
            for line in f:
                datas.append(json.loads(line))       
            print(f"###############  加载了 {len(datas)} 条数据  ###############  ")
        
        if not args.continue_generate:
            fout = open(args.output, 'w')
        else:
            fout = open(args.output, 'a')
            with open(args.output, 'r') as f:
                count = 0
                for line in f:
                    if line.strip():
                        count += 1
                datas = datas[count:]
                print(f"###############  继续从第 {count+1} 条数据生成  ###############  ")

        for data in tqdm(datas):
            # 获得req、prefix_code
            if '#include \"std_utils.h\"' in data['prompt']:
                idx = data['prompt'].index('#include \"std_utils.h\"')
                req = data['prompt'][:idx]
                prefix_code = data['prompt'][idx:]
            else:
                req = data['prompt']
                prefix_code = ''

            # RAG检索
            if args.use_rag:
                print('###############  正在进行RAG检索  ###############')
                rag_prompt = build_rag_prompt(req)
                print(rag_prompt)
                rag_results = get_topk(rag_prompt, args.rag_topk, args.rag_model)
                print(f'************** 已完成RAG检索，共{len(rag_results)}条，结果如下：**************\n{rag_results}\n')
            else:
                print('###############  未RAG检索  ###############')
                rag_results = []

            # 代码生成
            print('###############  正在进行代码生成  ###############')
            codegenerated_prompt = build_codegenerated_prompt(req, rag_results, prefix_code)
            print(codegenerated_prompt)
            generated_code = llm_generate_code(args.model, codegenerated_prompt)
            generated_code = extract_generated_code(generated_code)
            print(f'************** 已完成代码生成，结果如下：************** \n{generated_code}\n')

            # 代码编译
            if args.use_compile:
                print('###############  正在进行编译校正  ###############')
                generated_code = llm_complie(
                    args.model, 
                    req, 
                    prefix_code, 
                    generated_code,
                    rag_results,
                    ITERATION=args.llm_compile_iteration)
                print(f'************** 已完成编译校正，结果如下：************** \n{generated_code}\n')
            else:
                print('###############  未编译校正  ###############')

            # 自我校对
            if args.use_selfcorrect:
                print('###############  正在进行自我校正  ###############')
                generated_code = llm_selfcorrect(
                    args.model, 
                    req, 
                    prefix_code, 
                    generated_code, 
                    rag_results,
                    ITERATION=args.llm_selfcorrect_iteration,
                    w_compile=args.use_selfcorrect_w_compile)
                print(f'************** 已完成自我校正，结果如下：************** \n{generated_code}\n')
            else:
                print('###############  未自我校正  ###############')
            
            # 代码静态检测
            if args.use_static:
                print('###############  正在进行静态分析校正  ###############')
                generated_code = llm_static(
                    args.model, 
                    req, 
                    prefix_code, 
                    generated_code, 
                    rag_results,
                    ITERATION=args.llm_static_iteration, 
                    TOLERANCE=args.llm_static_tolerance,
                    w_compile=args.use_static_w_compile,
                    w_selfcorrect=args.use_static_w_selfcorrect)
                print(f'************** 已完成静态分析校正，结果如下：************** \n{generated_code}\n')
            else:
                print('###############  未静态分析校正  ###############')

            # 数据写入
            print('###############  正在进行数据写入  ###############')
            data['generation'] = extract_generated_code(generated_code)
            json.dump(data, fout, ensure_ascii=False)
            fout.write('\n')
            print('**************  已完成数据写入  **************\n')

        fout.close()
    except Exception as e:
        print(f"处理异常：{str(e)}")
        import traceback
        traceback.print_exc()
