import os
import numpy as np
import pickle
from typing import Optional, List, Dict
from config.settings import Config
from services.embedding_service import EmbeddingService

class ImageSearch:
    def __init__(self, mode: str = 'api', model_name: Optional[str] = None):
        self.embedding_service = EmbeddingService()
        self.embedding_service.set_mode(mode, model_name)
        self.image_data = None
        self._try_load_cache()
        
    def _try_load_cache(self) -> None:
        """尝试加载缓存"""
        cache_file = self._get_cache_file()
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                valid_embeddings = []
                for item in cached_data:
                    full_path = os.path.join(Config.IMAGE_DIR, item['filename'])
                    if os.path.exists(full_path):
                        valid_embeddings.append(item)
                    
                if valid_embeddings:
                    self.image_data = valid_embeddings
                    if len(valid_embeddings) != len(cached_data):
                        with open(cache_file, 'wb') as f:
                            pickle.dump(valid_embeddings, f)
                else:
                    self.image_data = None
            except (pickle.UnpicklingError, EOFError):
                self.image_data = None
                
    def _get_cache_file(self) -> str:
        """获取当前模式的缓存文件路径"""
        cache_key = f"{self.embedding_service.mode}"
        if self.embedding_service.mode == 'local' and self.embedding_service.selected_model:
            cache_key += f"_{self.embedding_service.selected_model}"
        return Config.CACHE_FILE.replace('.pkl', f'_{cache_key}.pkl')
        
    def set_mode(self, mode: str, model_name: Optional[str] = None) -> None:
        """切换搜索模式和模型"""
        try:
            self.embedding_service.set_mode(mode, model_name)
            self._try_load_cache()
        except Exception as e:
            print(f"模式切换失败: {str(e)}")
            # 保持错误状态，让UI层处理
            if mode == 'local':
                self.embedding_service.mode = mode
                self.embedding_service.selected_model = model_name
                self.embedding_service.current_model = None
        
    def download_model(self) -> None:
        """下载选中的模型"""
        self.embedding_service.download_selected_model()
        
    def load_model(self) -> None:
        """加载选中的模型"""
        self.embedding_service.load_selected_model()
    
    def has_cache(self) -> bool:
        """检查是否有可用的缓存"""
        return self.image_data is not None
    
    def generate_cache(self) -> None:
        """生成嵌入缓存"""
        if self.embedding_service.mode == 'local':
            self.load_model()  # 确保模型已加载
            
        if not os.path.exists(Config.IMAGE_DIR):
            os.makedirs(Config.IMAGE_DIR, exist_ok=True)
            
        # 获取图片文件
        image_files = [
            os.path.splitext(f)[0]
            for f in os.listdir(Config.IMAGE_DIR)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))  # 支持更多图片格式
        ]
        
        # 生成嵌入
        embeddings = []
        for filename in image_files:
            try:
                full_filename = None
                for ext in ['.png', '.jpg', '.jpeg', '.gif']:
                    if os.path.exists(os.path.join(Config.IMAGE_DIR, filename + ext)):
                        full_filename = filename + ext
                        break
                
                if full_filename:
                    embedding = self.embedding_service.get_embedding(filename)
                    embeddings.append({
                        "filename": full_filename,
                        "embedding": embedding
                    })
            except Exception as e:
                print(f"生成嵌入失败 [{filename}]: {str(e)}")
                
        # 保存缓存
        if embeddings:
            cache_file = self._get_cache_file()
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
            self.image_data = embeddings
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """余弦相似度计算"""
        return np.dot(a, b)
    
    def search(self, query: str, top_k: int = 5, api_key: Optional[str] = None) -> List[str]:
        """语义搜索最匹配的图片"""
        if not self.has_cache():
            return []
            
        try:
            query_embedding = self.embedding_service.get_embedding(query, api_key)
        except Exception as e:
            print(f"查询嵌入生成失败: {str(e)}")
            return []
        
        similarities = []
        for img in self.image_data:
            full_path = os.path.join(Config.IMAGE_DIR, img["filename"])
            if os.path.exists(full_path):  # 再次验证文件存在
                similarities.append((full_path, self._cosine_similarity(query_embedding, img["embedding"])))
        
        if not similarities:
            return []
            
        # 按相似度降序排序并返回前top_k个结果
        sorted_items = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        return [item[0] for item in sorted_items] 