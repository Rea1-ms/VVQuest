import os
import numpy as np
import pickle
from typing import Optional, List, Dict
from config.settings import config
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
                    # 获取文件路径
                    if 'filepath' in item:
                        full_path = item['filepath']
                    else:
                        full_path = os.path.join(config.get_absolute_image_dirs()[0], item['filename'])
                        # 添加filepath字段
                        item['filepath'] = full_path
                    
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
        return config.get_absolute_cache_file().replace('.pkl', f'_{cache_key}.pkl')
        
    def set_mode(self, mode: str, model_name: Optional[str] = None) -> None:
        """切换搜索模式和模型"""
        try:
            self.embedding_service.set_mode(mode, model_name)
            # 清空当前缓存
            self.image_data = None
            # 尝试加载新模式/模型的缓存
            self._try_load_cache()
        except Exception as e:
            print(f"模式切换失败: {str(e)}")
            # 保持错误状态，让UI层处理
            if mode == 'local':
                self.embedding_service.mode = mode
                self.embedding_service.selected_model = model_name
                self.embedding_service.current_model = None
            # 确保清空缓存
            self.image_data = None
        
    def download_model(self) -> None:
        """下载选中的模型"""
        self.embedding_service.download_selected_model()
        
    def load_model(self) -> None:
        """加载选中的模型"""
        self.embedding_service.load_selected_model()
    
    def has_cache(self) -> bool:
        """检查是否有可用的缓存"""
        return self.image_data is not None
    
    def generate_cache(self, progress_bar) -> None:
        """生成缓存"""
        if self.embedding_service.mode == 'local':
            self.load_model()  # 确保模型已加载
            
        # 获取所有图片目录
        image_dirs = config.get_absolute_image_dirs()
        for img_dir in image_dirs:
            if not os.path.exists(img_dir):
                os.makedirs(img_dir, exist_ok=True)

        self._try_load_cache()
        generated_files = []
        if self.image_data is not None:
            # 确保所有缓存数据都有filepath字段
            for item in self.image_data:
                if 'filepath' not in item:
                    item['filepath'] = os.path.join(config.get_absolute_image_dirs()[0], item['filename'])
            generated_files = [i['filepath'] for i in self.image_data]

        # 获取所有路径
        all_dir = []
        for img_dir in image_dirs:
            all_dir.extend([entry.path for entry in os.scandir(img_dir)])

        # 获取图片文件
        image_files = [
            f
            for f in all_dir
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))  # 支持更多图片格式
        ]
        
        # 生成缓存
        embeddings = []
        errors = []  # 收集错误
        error_count = 0  # 错误计数
        length = len(image_files)
        for index, filepath in enumerate(image_files):
            try:
                filename = os.path.splitext(os.path.basename(filepath))[0]

                full_filename = None
                for ext in ['.png', '.jpg', '.jpeg', '.gif']:
                    if os.path.exists(os.path.join(os.path.dirname(filepath), filename + ext)):
                        full_filename = filename + ext
                        break
                
                if full_filename:
                    if filepath in generated_files:
                        # 使用已经存在的缓存
                        embedding = self.image_data[generated_files.index(filepath)]['embedding']
                    else:
                        embedding = self.embedding_service.get_embedding(filename)
                    embeddings.append({
                        "filename": full_filename,
                        "filepath": filepath,
                        "embedding": embedding
                    })

                progress_bar.progress((index + 1) / length, text=f"处理图片 {index + 1}/{length}")
            except Exception as e:
                error_count += 1
                error_msg = f"错误{error_count}: 缓存生成失败 [{filepath}]: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
                if error_count >= 3:
                    error_summary = f"连续遇到错误：\n" + "\n".join(errors)
                    print(error_summary)
                    raise RuntimeError(error_summary)
                
        # 如果有错误发生但未达到停止阈值，仍然抛出异常
        if errors:
            error_summary = "\n".join(errors)
            print(error_summary)
            raise RuntimeError(error_summary)
                
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
            if 'filepath' not in img and config.misc.adapt_for_old_version:
                img['filepath'] = os.path.join(config.get_absolute_image_dirs()[0], img["filename"])
            if os.path.exists(img['filepath']):
                similarities.append((img['filepath'], self._cosine_similarity(query_embedding, img["embedding"])))
        
        if not similarities:
            return []
            
        # 按相似度降序排序并返回前top_k个结果
        sorted_items = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        return [item[0] for item in sorted_items] 