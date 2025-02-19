import os
import numpy as np
import pickle
import re
from typing import Optional, List, Dict
from config.settings import Config, UIConfig
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
                    if 'filepath' in item.keys():
                        full_path = item['filepath']
                    else:
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
    
    def generate_cache(self, progress_bar) -> None:
        """生成嵌入缓存"""
        if self.embedding_service.mode == 'local':
            self.load_model()  # 确保模型已加载
            
        if not os.path.exists(Config.IMAGE_DIR):
            os.makedirs(Config.IMAGE_DIR, exist_ok=True)

        self._try_load_cache()
        generated_files = []
        if self.image_data is not None:
            generated_files = [i['filepath'] for i in self.image_data]

        # 获取所有路径
        def get_all_file_paths(folder_path):
            # 用于存储所有文件的绝对路径
            file_paths = []
            # 使用os.walk()遍历文件夹及其子文件夹
            for root, directories, files in os.walk(folder_path):
                for filename in files:
                    # 构建文件的绝对路径
                    file_path = os.path.join(root, filename)
                    # 将绝对路径添加到列表中
                    file_paths.append(file_path)
            return file_paths
        all_dir = []
        for img_dir in [v['path'] for v in UIConfig().image_dirs.values()]:
            if not os.path.isabs(img_dir): img_dir = os.path.join(Config.BASE_DIR, img_dir)
            all_dir.extend(get_all_file_paths(img_dir))

        # 获取图片文件
        image_files = [
            f
            for f in all_dir
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))  # 支持更多图片格式
        ]
        
        # 生成嵌入
        embeddings = []
        length = len(image_files)
        for dirs_k, dirs_v in UIConfig().image_dirs.items():
            if 'regex' in dirs_v.keys():
                replace_patterns_regex = {dirs_v['regex']['pattern']: dirs_v['regex']['replacement']}
            else:
                replace_patterns_regex = None

            image_type = dirs_v.setdefault('type', 'None')

            for index, filepath in enumerate(image_files):
                try:
                    if not os.path.isabs(filepath): filepath = os.path.join(Config.BASE_DIR, filepath)
                    filename = os.path.splitext(os.path.basename(filepath))[0]

                    full_filename = None
                    for ext in ['.png', '.jpg', '.jpeg', '.gif']:
                        if os.path.exists(os.path.join(os.path.dirname(filepath), filename + ext)):
                            full_filename = filename + ext
                            break

                    if full_filename:
                        if filepath in generated_files:
                            # 使用已经存在的embedding
                            embedding = self.image_data[generated_files.index(filepath)]['embedding']
                            embedding_name = self.image_data[generated_files.index(filepath)]['embedding_name']
                        else:
                            embedding_name = filename
                            if replace_patterns_regex is not None:
                                for pattern, replacement in replace_patterns_regex.items():
                                    embedding_name = re.sub(pattern, replacement, embedding_name)
                            embedding = self.embedding_service.get_embedding(embedding_name)
                        embeddings.append({
                            "filename": full_filename,
                            "filepath": filepath,
                            "embedding": embedding,
                            "embedding_name": embedding_name,
                            "type": image_type if image_type is not None else 'Normal'
                        })

                    progress_bar.progress((index + 1) / length, text=f"处理图片 {index + 1}/{length}")
                except Exception as e:
                    print(f"生成嵌入失败 [{filepath}]: {str(e)}")
                
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
            if 'filepath' not in img.keys() and Config.ADAPT_FOR_OLD_VERSION:
                full_path = os.path.join(Config.IMAGE_DIR, img["filename"])
            else:
                full_path = img['filepath']
            if os.path.exists(full_path):  # 再次验证文件存在
                similarities.append((full_path, self._cosine_similarity(query_embedding, img["embedding"])))
        
        if not similarities:
            return []
            
        # 按相似度降序排序并返回前top_k个结果
        sorted_items = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        return [item[0] for item in sorted_items] 