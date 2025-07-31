import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# API Keys
SILICONFLOW_API_KEY = os.getenv('SiliconFlow_API_KEY')
DEEPINFRA_API_KEY = os.getenv('DeepInfra_API_KEY')
GOOGLE_API_KEY = os.getenv('Google_API_KEY')
AZURE_API_KEY = os.getenv('AZURE_API_KEY')

# API Base URLs
SILICONFLOW_API_BASE = "https://api.siliconflow.cn/v1"
DEEPINFRA_API_BASE = "https://api.deepinfra.com/v1/openai"
GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

AZURE_API_BASE = os.getenv('AZURE_API_BASE')
AZURE_API_VERSION = os.getenv('AZURE_API_VERSION')

# Azure Deployments
AZURE_DEPLOYMENT_GPT4O = os.getenv('AZURE_DEPLOYMENT_GPT4O')
AZURE_DEPLOYMENT_GPT4O_MINI = os.getenv('AZURE_DEPLOYMENT_GPT4O_MINI')
AZURE_DEPLOYMENT_GPT35 = os.getenv('AZURE_DEPLOYMENT_GPT35')

# 可用的模型列表
AVAILABLE_MODELS = {
    # AZURE Models
    'gpt-4o': {
        'provider': 'azure',
        'model_name': 'gpt-4o',
        'deployment_name': AZURE_DEPLOYMENT_GPT4O
    },
    'gpt-4o-mini': {
        'provider': 'azure',
        'model_name': 'gpt-4o-mini',
        'deployment_name': AZURE_DEPLOYMENT_GPT4O_MINI
    },
    # SILICONFLOW_API
    'deepseek-v2.5': {
        'provider': 'siliconflow',
        'model_name': 'deepseek-ai/DeepSeek-V2.5'
    },
    'qwen-32b': {
        'provider': 'siliconflow',
        'model_name': 'Qwen/QwQ-32B'
    },
    'qwen3-30b': {
        'provider': 'siliconflow',
        'model_name': 'Qwen/Qwen3-30B-A3B'
    },
    'qwen3-235b': {
        'provider': 'siliconflow',
        'model_name': 'Qwen/Qwen3-235B-A22B'
    },
    'llama-70b': {
        'provider': 'siliconflow',
        'model_name': 'meta-llama/Llama-3.3-70B-Instruct'
    },
    'qwen-72b': {
        'provider': 'siliconflow',
        'model_name': 'Qwen/QVQ-72B-Preview'
    },
    'glm-4-9b-chat': {
        'provider': 'siliconflow',
        'model_name': 'THUDM/glm-4-9b-chat'
    },
    #DEEPINFRA_API
    'deepseek-r1': {
        'provider': 'deepinfra',
        'model_name': 'deepseek-ai/DeepSeek-R1'
    },
    'deepseek-v3': {
        'provider': 'deepinfra',
        'model_name': 'deepseek-ai/DeepSeek-V3'
    },
    'deepseek-r1-distill-llama-70b': {
        'provider': 'deepinfra',
        'model_name': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'
    }
}

# 当前使用的配置
current_config = {
    'api_key': None,
    'api_base': None,
    'model_name': None,
    'provider': None,
    'azure_config': {
        'api_version': None,
        'deployment_name': None
    }
}

def set_current_config(provider='openai'):
    """
    设置当前使用的API配置
    :param provider: 可选值 'siliconflow', 'deepinfra',  'azure', 'google'
    """
    provider = provider.lower()
    current_config['provider'] = provider
    
    if provider == 'siliconflow':
        current_config['api_key'] = SILICONFLOW_API_KEY
        current_config['api_base'] = SILICONFLOW_API_BASE
    elif provider == 'deepinfra':
        current_config['api_key'] = DEEPINFRA_API_KEY
        current_config['api_base'] = DEEPINFRA_API_BASE
    elif provider == 'azure':
        current_config['api_key'] = AZURE_API_KEY
        current_config['api_base'] = AZURE_API_BASE
        current_config['azure_config']['api_version'] = AZURE_API_VERSION
        current_config['azure_config']['deployment_name'] = AZURE_DEPLOYMENT_GPT4O
    elif provider == 'google':
        current_config['api_key'] = GOOGLE_API_KEY
        current_config['api_base'] = GOOGLE_API_BASE
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def set_model(model_name):
    """
    设置当前使用的模型
    :param model_name: 模型名称，必须在AVAILABLE_MODELS中存在
    """
    if model_name in AVAILABLE_MODELS:
        model_info = AVAILABLE_MODELS[model_name]
        current_config['model_name'] = model_info['model_name']
        set_current_config(model_info['provider'])
        
        # 如果是Azure模型，设置对应的deployment_name
        if model_info['provider'] == 'azure':
            current_config['azure_config']['deployment_name'] = model_info['deployment_name']
    else:
        raise ValueError(f"Unsupported model: {model_name}") 