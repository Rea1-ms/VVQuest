import os
import requests
from config.settings import Config
from typing import List, Optional, Union
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from huggingface_hub import snapshot_download
from tqdm import tqdm

class EmbeddingService:
    def __init__(self):
        self.api_key = Config.SILICON_API_KEY
        self.endpoint = "https://api.siliconflow.com/v1/embeddings" 
        self.local_models = {}
        self.current_model = None
        self.mode = 'api'  # 'api' or 'local'
        self.selected_model = None
        
    def _download_model(self, model_name: str) -> None:
        """下载模型到本地"""
        model_info = Config.EMBEDDING_MODELS.get(model_name)
        if not model_info:
            raise ValueError(f"未知的模型: {model_name}")
            
        model_path = Config.get_model_path(model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
            print(f"正在下载模型 {model_name}...")
            snapshot_download(
                repo_id=model_info['name'],
                local_dir=model_path,
                local_dir_use_symlinks=False
            )
            
    def _load_local_model(self, model_name: str) -> None:
        """加载本地模型"""
        try:
            if model_name not in self.local_models:
                model_path = Config.get_model_path(model_name)
                if not os.path.exists(model_path):
                    raise RuntimeError(f"模型 {model_name} 尚未下载")
                print(f"正在加载模型 {model_name}...")
                self.local_models[model_name] = BGEM3FlagModel(
                    model_path,
                    use_fp16=True
                )
            self.current_model = self.local_models[model_name]
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            self.current_model = None
            # 如果加载失败，从缓存中移除
            if model_name in self.local_models:
                del self.local_models[model_name]
            # 删除可能损坏的模型文件
            model_path = Config.get_model_path(model_name)
            if os.path.exists(model_path):
                import shutil
                shutil.rmtree(model_path)
            raise RuntimeError(f"模型加载失败，请重新下载模型。错误信息: {str(e)}")
            
    def set_mode(self, mode: str, model_name: Optional[str] = None) -> None:
        """设置服务模式(api/local)和选择模型"""
        if mode not in ['api', 'local']:
            raise ValueError("模式必须是 'api' 或 'local'")
            
        self.mode = mode
        if mode == 'local':
            if model_name is None:
                model_name = Config.DEFAULT_MODEL
            self.selected_model = model_name
            # 如果模型已下载，尝试加载
            if self.is_model_downloaded(model_name):
                try:
                    self._load_local_model(model_name)
                except Exception as e:
                    print(f"模型加载失败: {str(e)}")
                    self.current_model = None
            else:
                self.current_model = None
        else:
            self.current_model = None
            self.selected_model = None
            
    def download_selected_model(self) -> None:
        """下载已选择的模型"""
        if self.mode != 'local' or not self.selected_model:
            raise RuntimeError("请先选择本地模式和模型")
        self._download_model(self.selected_model)
        # 下载后自动加载
        self._load_local_model(self.selected_model)
            
    def load_selected_model(self) -> None:
        """加载已选择的模型"""
        if self.mode != 'local' or not self.selected_model:
            raise RuntimeError("请先选择本地模式和模型")
        self._load_local_model(self.selected_model)
            
    def is_model_downloaded(self, model_name: str) -> bool:
        """检查模型是否已下载"""
        model_path = Config.get_model_path(model_name)
        return os.path.exists(model_path)
            
    @staticmethod
    def normalize_embedding(embedding: Union[List[float], np.ndarray]) -> np.ndarray:
        """归一化嵌入向量"""
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        return embedding / np.linalg.norm(embedding)
    
    def get_embedding(self, text: str, key: str = None) -> np.ndarray:
        """获取文本嵌入并归一化"""
        if self.mode == 'api':
            # API 模式
            headers = {"Authorization": f"Bearer {key if key is not None else self.api_key}"}
            payload = {
                "input": text,
                "model": Config.EMBEDDING_MODELS['bge-m3']['name'],
                "encoding_format": "float"  # 指定返回格式
            }
            try:
                response = requests.post(self.endpoint, json=payload, headers=headers)
                response.raise_for_status()  # 抛出详细的HTTP错误
                embedding = response.json()['data'][0]['embedding']
            except requests.exceptions.RequestException as e:
                if hasattr(e.response, 'status_code') and e.response.status_code == 400:
                    # 尝试打印详细的错误信息
                    error_msg = e.response.json() if e.response.text else "未知错误"
                    print(f"API请求参数错误: {error_msg}")
                raise RuntimeError(f"API请求失败: {str(e)}\n请求参数: {payload}")
        else:
            # 本地模式
            if self.current_model is None:
                # 如果模型未加载但已下载，尝试加载
                if self.selected_model and self.is_model_downloaded(self.selected_model):
                    self._load_local_model(self.selected_model)
                else:
                    raise RuntimeError("未加载本地模型")
            # 每次都重新计算嵌入向量
            output = self.current_model.encode(
                text, 
                return_dense=True, 
                return_sparse=False, 
                return_colbert_vecs=False
            )
            embedding = output['dense_vecs']
                
        # 确保返回新的归一化向量
        return self.normalize_embedding(embedding.copy() if isinstance(embedding, np.ndarray) else embedding)