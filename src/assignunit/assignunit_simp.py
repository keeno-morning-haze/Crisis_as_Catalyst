import csv
import time
import openai
import random
from collections import defaultdict
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import sys
import os
import json

# 获取主文件夹路径
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(main_dir)

import api

# ========== 系统提示词模板 ==========
system_template = """
您是一位危机管理领域的专家，专注于组织间协作与决策，具备深厚的公共管理、危机协同理论和实践经验。
您了解警察、消防、民事应急和军队等组织的协作方式。
接下来你需要完成一项问卷实验，模拟真实场景完成决策任务。依次会有四个不同背景的问卷，请区别你在不同问卷下的身份设定。
"""

# ========== 用户提示词模板 ==========
def get_user_template(scenario):
    base_template = """
    这是场景{scenario}下的问卷，请勿与其他场景混淆。
    你是代表蓝色小队的指挥官，需要与另一个红色小队协作来应对一场危机。下面是你的身份设定：
    {familiarity_text}
    {ecfi_text}
    
    共12个单位进行本次危机响应，其中6个属于蓝色小队，6个属于红色小队。
    用数字表示单位编号，1-6 为蓝色小队的单位，7-12 为红色小队的单位。
    每天只能派遣一个单位，第二到七天单位无法进入危机区域，因此您只能为第一天以及第八天到第十八天分配单位。
    你需要决定各单位到达机场的顺序，以实现最优的危机响应。
    
    请你一步一步思考，考虑到各种因素的影响，并给出你的决策。
    格式：你的回答最后一行只能是12个数字，分别代表第一天、第八天到第十八天派出的队伍编号，用逗号','分隔开。每个单位编号只出现一次。不要有其他说明性文字。
    """

    familiarity_dict = {
        "A": "你不了解红色小队。", # 低熟悉度
        "B": "你了解红色小队。", # 高熟悉度
        "C": "你不了解红色小队。", # 低熟悉度
        "D": "你了解红色小队。", # 高熟悉度
    }
    ecfi_dict = {
        "A": "你不期望未来再次与红色小队协作。", # 低预期
        "B": "你不期望未来再次与红色小队协作。", # 低预期
        "C": "你期望未来再次与红色小队协作。", # 高预期
        "D": "你期望未来再次与红色小队协作。", # 高预期
    }
    return base_template.format(
        familiarity_text=familiarity_dict[scenario],
        ecfi_text=ecfi_dict[scenario],
        scenario=scenario
    )

# 追问提示词模板
def get_follow_up_template(scenario):
    return f"""
    关于你刚才回答的第一个问卷，请描述你做决定背后的主要指导原则，语言请尽量简明扼要。
    """

# ========== 构建 LLM ==========
def build_agent_llm():
    """shi
    根据当前配置构建LLM
    """
    if api.current_config['provider'] == 'azure':
        llm = AzureChatOpenAI(
            azure_endpoint=api.current_config['api_base'],
            api_key=api.current_config['api_key'],
            api_version=api.current_config['azure_config']['api_version'],
            deployment_name=api.current_config['azure_config']['deployment_name'],
            temperature=0.7,
            top_p=1.0, 
            timeout=30,
            max_tokens=4096
        )
    else:
        # 对于其他提供商（siliconflow, deepinfra），使用他们的基地址
        openai.api_base = api.current_config['api_base']
        openai.api_key = api.current_config['api_key']
        
        llm = ChatOpenAI(
            api_key=api.current_config['api_key'],
            model_name=api.current_config['model_name'],
            temperature=0.7,
            top_p=1.0,
            presence_penalty=0,
            frequency_penalty=0,
            max_tokens=4096,
            openai_api_base=api.current_config['api_base'],
            openai_api_key=api.current_config['api_key']
        )
    return llm

# ========== 生成问卷回答并保存 ==========
def generate_responses(model_name, n, raw_file):
    """
    生成指定模型的问卷回答并追加到文件
    :param model_name: 模型名称
    :param n: 生成数量
    :param raw_file: 结果文件路径
    """
    scenarios = ["A", "B", "C", "D"]
    
    # 设置模型
    api.set_model(model_name)
    
    # 更新openai配置
    openai.api_key = api.current_config['api_key']
    openai.api_base = api.current_config['api_base']
    
    # 获取当前已有的test_id数量
    current_id = 0
    if os.path.exists(raw_file):
        with open(raw_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['model'] == model_name:
                    try:
                        current_id = max(current_id, int(row['test_id']))
                    except ValueError:
                        continue
    
    # 准备motivation文件
    motivation_file = raw_file.rsplit('.', 1)[0] + '_motivation.json'
    motivations = []
    if os.path.exists(motivation_file):
        with open(motivation_file, 'r', encoding='utf-8') as f:
            try:
                motivations = json.load(f)
            except json.JSONDecodeError:
                motivations = []
    
    # 检查文件是否存在，如果不存在则创建并写入表头
    file_exists = os.path.exists(raw_file)
    if not file_exists:
        with open(raw_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['model', 'scenario'] + ["Day1"] + [f"Day{day}" for day in range(8,19)] + ['test_id']
            writer.writerow(header)
    
    # 追加模式打开文件
    with open(raw_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        for i in range(n):
            # 构建LLM
            llm = build_agent_llm()
            
            # 初始化对话历史
            messages = [
                SystemMessage(content=system_template)
            ]
            
            # 随机排序场景顺序
            shuffled_scenarios = scenarios.copy()
            random.shuffle(shuffled_scenarios)
            
            # 记录第一个场景，用于后续追问
            first_scenario = shuffled_scenarios[0]
            
            # 创建一个字典来存储所有场景的回答
            scenario_responses = {}
            
            # 生成测试ID
            test_id = f"{current_id+i+1}"
            
            retry_count = 0
            max_retries = 3
            
            print(f"开始测试 {test_id}，场景顺序: {shuffled_scenarios}")
            
            # 设置随机种子
            seed = int(time.time() * 1000) + int(test_id) + random.randint(0, 1000000)
            random.seed(seed)
            
            # 第一阶段：依次测试四个场景
            for idx, scenario in enumerate(shuffled_scenarios):
                success = False
                retry_count = 0
                
                while not success and retry_count < max_retries:
                    try:
                        print(f"测试 {test_id} 场景 {scenario} (第{idx+1}轮)...")
                        
                        # 添加当前场景的用户消息
                        messages.append(HumanMessage(content=get_user_template(scenario)))
                        
                        # 获取LLM回复
                        response = llm.invoke(messages)
                        messages.append(response)
                        
                        output_text = response.content
                        # 提取回复中的数字序列
                        output_text = [line.strip() for line in output_text.split('\n') if line.strip()][-1]
                        choices = [choice.strip() for choice in output_text.split(',')]
                        
                        if (len(choices) == 12 and 
                            all(choice.isdigit() and 1 <= int(choice) <= 12 for choice in choices) and 
                            len(set(choices)) == 12):  # 确保无重复编号
                            
                            # 存储场景回答
                            scenario_responses[scenario] = {
                                'test_order': idx + 1,
                                'choices': choices
                            }
                            
                            # 写入CSV
                            writer.writerow([
                                model_name,
                                scenario,
                                *choices,
                                test_id
                            ])
                            
                            success = True
                            time.sleep(random.uniform(1, 3))
                        else:
                            print(f"测试 {test_id} 场景 {scenario} 格式不正确，重试：{output_text}")
                            # 移除最后一条消息，重试
                            messages.pop()
                            retry_count += 1
                            
                    except Exception as e:
                        print(f"测试 {test_id} 场景 {scenario} 出错：{str(e)}")
                        # 移除最后一条消息，重试
                        if len(messages) > 1 and isinstance(messages[-1], HumanMessage):
                            messages.pop()
                        retry_count += 1
                
                # 如果重试仍然失败，跳过此场景
                if not success:
                    print(f"测试 {test_id} 场景 {scenario} 已重试{retry_count}次，跳过...")
                    writer.writerow([
                        model_name,
                        scenario,
                        *(['ERROR'] * 12),
                        test_id
                    ])
            
            # 第二阶段：追问第一个场景的决策动机
            success = False
            retry_count = 0
            
            while not success and retry_count < max_retries:
                try:
                    print(f"追问测试 {test_id} 场景 {first_scenario} 的决策动机...")
                    
                    # 添加追问消息
                    messages.append(HumanMessage(content=get_follow_up_template(first_scenario)))
                    
                    # 获取LLM回复
                    response = llm.invoke(messages)
                    messages.append(response)
                    
                    motivation_text = response.content
                    
                    # 存储决策动机到JSON
                    motivations.append({
                        'model': model_name,
                        'test_id': test_id,
                        'scenario': first_scenario,
                        'motivation': motivation_text
                    })
                    
                    success = True
                    
                except Exception as e:
                    print(f"追问测试 {test_id} 场景 {first_scenario} 出错：{str(e)}")
                    # 移除最后一条消息，重试
                    if len(messages) > 1 and isinstance(messages[-1], HumanMessage):
                        messages.pop()
                    retry_count += 1
                    time.sleep(random.uniform(3, 8))
            
            # 如果追问失败
            if not success:
                motivations.append({
                    'model': model_name,
                    'test_id': test_id,
                    'scenario': first_scenario,
                    'motivation': 'ERROR_RETRIEVING_MOTIVATION'
                })
            
            # 每完成一次完整测试，保存文件
            f.flush()
            # 保存motivation文件
            with open(motivation_file, 'w', encoding='utf-8') as mf:
                json.dump(motivations, mf, ensure_ascii=False, indent=2)
            
            print(f"完成测试 {test_id}")

# ========== 主程序入口 ==========
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='生成单位分配问卷回答')
    parser.add_argument('--model', type=str, default="qwen3-235b", help='要使用的模型名称')
    parser.add_argument('--num', type=int, default=10, help='要生成的测试次数')
    parser.add_argument('--output', type=str, default="data/assignunit/raw_responses_assignunit_simpcot.csv", help='输出文件路径')
    
    args = parser.parse_args()
    
    print(f"开始使用模型 {args.model} 生成 {args.num} 次测试...")
    generate_responses(args.model, args.num, args.output)
    print("\n生成完成！")
