"""
缓存工具模块，提供嵌入向量缓存功能
"""
import os
import json
import hashlib
import pickle
from typing import Dict, List, Any, Optional

class EmbeddingCache:
    """嵌入向量缓存管理器"""
    
    def __init__(self, cache_dir: str):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_key(self, text: str) -> str:
        """从文本内容生成缓存键"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, key: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{key}.pkl")
    
    def get(self, text: str) -> Optional[List[float]]:
        """
        从缓存获取嵌入向量
        
        Args:
            text: 文本内容
            
        Returns:
            缓存的嵌入向量，未找到则返回None
        """
        key = self._get_cache_key(text)
        cache_path = self._get_cache_path(key)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                # 缓存文件损坏，忽略
                return None
        
        return None
    
    def set(self, text: str, embedding: List[float]) -> None:
        """
        将嵌入向量保存到缓存
        
        Args:
            text: 文本内容
            embedding: 嵌入向量
        """
        key = self._get_cache_key(text)
        cache_path = self._get_cache_path(key)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)
    
    def clear(self) -> None:
        """清空缓存"""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                os.remove(os.path.join(self.cache_dir, filename))


class ProcessingStateManager:
    """断点续传状态管理器"""
    
    def __init__(self, state_file: str):
        """
        初始化状态管理器
        
        Args:
            state_file: 状态文件路径
        """
        self.state_file = state_file
        self.state_dir = os.path.dirname(state_file)
        os.makedirs(self.state_dir, exist_ok=True)
        
    def save_state(self, state: Dict[str, Any]) -> None:
        """
        保存处理状态
        
        Args:
            state: 状态数据
        """
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """
        加载处理状态
        
        Returns:
            状态数据，文件不存在则返回None
        """
        if not os.path.exists(self.state_file):
            return None
        
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            # 状态文件损坏，忽略
            return None
    
    def clear_state(self) -> None:
        """清除状态文件"""
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
