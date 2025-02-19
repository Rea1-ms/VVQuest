import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API 配置
    SILICON_API_KEY = os.getenv("SILICON_API_KEY")
    
    # 模型配置
    EMBEDDING_MODELS = {
        'bge-m3': {
            'name': 'BAAI/bge-m3',
            'size': '1.7GB',
            'performance': '高',
            'description': '高性能多语言模型'
        },
        'bge-large-zh-v1.5': {
            'name': 'BAAI/bge-large-zh-v1.5',
            'size': '1.2GB', 
            'performance': '中',
            'description': '中文优化模型'
        },
        'bge-small-zh-v1.5': {
            'name': 'BAAI/bge-small-zh-v1.5',
            'size': '400MB',
            'performance': '低',
            'description': '轻量级中文模型'
        }
    }
    
    DEFAULT_MODEL = 'bge-large-zh-v1.5'
    
    # 路径配置
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    IMAGE_DIR = os.path.join(BASE_DIR, 'data/images')
    IMAGE_DIRS = [os.path.join(BASE_DIR, 'data/images'), os.path.join(BASE_DIR, 'data/images_vv2')]
    CACHE_FILE = os.path.join(BASE_DIR, 'data/embeddings.pkl')
    MODELS_DIR = os.path.join(BASE_DIR, 'data/models')
    ADAPT_FOR_OLD_VERSION = True
    
    @classmethod
    def get_model_path(cls, model_name: str) -> str:
        """获取模型保存路径"""
        return os.path.join(cls.MODELS_DIR, model_name.replace('/', '_')) 