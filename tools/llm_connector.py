"""
MiniMax LLM连接器
提供与MiniMax-M2.7模型的API交互能力
"""
import os
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests


class MiniMaxConnector:
    """
    MiniMax API连接器

    功能：
    - 与MiniMax-M2.7模型进行对话
    - 管理API认证和请求
    - 处理响应和错误
    - 实现请求缓存和重试机制
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.minimax.chat/v1",
        model: str = "MiniMax-M2.7",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        初始化MiniMax连接器

        Args:
            api_key: API密钥（如果为None，从环境变量读取）
            base_url: API基础URL
            model: 模型名称
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.logger = logging.getLogger("MiniMaxConnector")

        # API配置
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY", "")
        if not self.api_key:
            self.logger.warning("⚠️ MiniMax API密钥未设置，请设置MINIMAX_API_KEY环境变量")

        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        # 会话管理
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        })

        # 统计信息
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens_used = 0
        self.cache_hits = 0

        # 简单缓存（生产环境建议使用Redis）
        self.response_cache: Dict[str, Any] = {}
        self.max_cache_size = 1000

        self.logger.info(f"✅ MiniMax连接器初始化完成 - 模型: {model}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        执行对话补全

        Args:
            messages: 消息列表，格式为 [{"role": "user/assistant/system", "content": "..."}]
            temperature: 温度参数 (0.0-1.0)
            top_p: 核采样参数 (0.0-1.0)
            max_tokens: 最大生成token数
            use_cache: 是否使用缓存

        Returns:
            包含响应文本和元数据的字典
        """
        start_time = time.time()
        self.total_requests += 1

        try:
            # 检查缓存
            cache_key = self._generate_cache_key(messages, temperature, top_p, max_tokens)
            if use_cache and cache_key in self.response_cache:
                self.cache_hits += 1
                cached_response = self.response_cache[cache_key].copy()
                cached_response['from_cache'] = True
                self.logger.debug(f"💾 缓存命中")
                return cached_response

            # 构建请求体
            request_body = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": False
            }

            # 发送请求（带重试）
            response_data = self._send_request_with_retry(request_body)

            # 解析响应
            result = self._parse_response(response_data, start_time)

            # 缓存结果
            if use_cache and len(self.response_cache) < self.max_cache_size:
                self.response_cache[cache_key] = result.copy()

            self.successful_requests += 1
            self.logger.info(
                f"✅ 请求成功 - 耗时: {result['response_time_ms']:.0f}ms, "
                f"tokens: {result.get('usage', {}).get('total_tokens', 'N/A')}"
            )

            return result

        except Exception as e:
            self.failed_requests += 1
            error_msg = f"❌ MiniMax API请求失败: {str(e)}"
            self.logger.error(error_msg)

            return {
                "success": False,
                "error": str(e),
                "response_text": "",
                "response_time_ms": (time.time() - start_time) * 1000,
                "from_cache": False
            }

    def _send_request_with_retry(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """发送请求并自动重试"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                url = f"{self.base_url}/text/chatcompletion_v2"

                response = self.session.post(
                    url,
                    json=request_body,
                    timeout=self.timeout
                )

                # 检查HTTP状态码
                response.raise_for_status()

                return response.json()

            except requests.exceptions.Timeout:
                last_error = f"请求超时（尝试 {attempt + 1}/{self.max_retries}）"
                self.logger.warning(f"⏱️ {last_error}")

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if hasattr(e, 'response') else 0
                last_error = f"HTTP错误 {status_code}: {str(e)}"
                self.logger.error(f"❌ {last_error}")

                # 4xx错误通常不需要重试
                if 400 <= status_code < 500:
                    raise

            except requests.exceptions.ConnectionError:
                last_error = f"连接错误（尝试 {attempt + 1}/{self.max_retries}）"
                self.logger.warning(f"🔌 {last_error}")

            except Exception as e:
                last_error = f"未知错误: {str(e)}"
                self.logger.error(f"❌ {last_error}")

            # 等待后重试
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                self.logger.info(f"🔄 {wait_time}秒后重试...")
                time.sleep(wait_time)

        raise Exception(f"所有重试均失败。最后错误: {last_error}")

    def _parse_response(self, response_data: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """解析API响应"""
        response_time_ms = (time.time() - start_time) * 1000

        # 提取响应文本
        choices = response_data.get("choices", [])
        if not choices:
            raise Exception("响应中没有choices字段")

        message = choices[0].get("message", {})
        response_text = message.get("content", "")

        # 提取使用情况
        usage = response_data.get("usage", {})
        self.total_tokens_used += usage.get("total_tokens", 0)

        return {
            "success": True,
            "response_text": response_text,
            "response_time_ms": response_time_ms,
            "usage": usage,
            "model": response_data.get("model", self.model),
            "finish_reason": choices[0].get("finish_reason", "unknown"),
            "from_cache": False,
            "timestamp": datetime.now().isoformat()
        }

    def _generate_cache_key(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int
    ) -> str:
        """生成缓存键"""
        import hashlib
        import json

        key_data = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        success_rate = (
            self.successful_requests / self.total_requests * 100
            if self.total_requests > 0 else 0
        )

        cache_hit_rate = (
            self.cache_hits / self.total_requests * 100
            if self.total_requests > 0 else 0
        )

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(success_rate, 2),
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(cache_hit_rate, 2),
            "total_tokens_used": self.total_tokens_used,
            "cache_size": len(self.response_cache)
        }

    def clear_cache(self):
        """清空缓存"""
        self.response_cache.clear()
        self.logger.info("🗑️ 缓存已清空")

    def close(self):
        """关闭连接器"""
        self.session.close()
        self.logger.info("👋 MiniMax连接器已关闭")