"""
API连接器 - API Connector
提供统一、安全的API调用接口
支持多种认证方式、请求重试和速率限制
专为CPU环境优化，轻量级且可靠的API管理解决方案
"""
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class AuthType(Enum):
    """认证类型"""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"


class HttpMethod(Enum):
    """HTTP方法"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass
class APIConfig:
    """API配置"""
    base_url: str
    auth_type: AuthType = AuthType.NONE
    api_key: Optional[str] = None
    bearer_token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    timeout_seconds: float = 10.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    rate_limit_requests_per_minute: int = 60

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "base_url": self.base_url,
            "auth_type": self.auth_type.value,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "rate_limit_rpm": self.rate_limit_requests_per_minute
        }


@dataclass
class APIResponse:
    """API响应"""
    success: bool
    status_code: int
    data: Any
    headers: Dict[str, str] = field(default_factory=dict)
    response_time_ms: float = 0.0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "status_code": self.status_code,
            "data": self.data,
            "headers": self.headers,
            "response_time_ms": round(self.response_time_ms, 2),
            "error_message": self.error_message
        }


@dataclass
class RequestLog:
    """请求日志"""
    timestamp: datetime
    method: str
    url: str
    status_code: int
    response_time_ms: float
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "method": self.method,
            "url": self.url,
            "status_code": self.status_code,
            "response_time_ms": round(self.response_time_ms, 2),
            "success": self.success,
            "error": self.error
        }


class APIConnector:
    """
    API连接器

    功能：
    - 统一接口：提供一致的API调用接口
    - 多种认证：支持API Key、Bearer Token、Basic Auth等
    - 自动重试：失败时自动重试请求
    - 速率限制：控制请求频率避免被封
    - 错误处理：完善的错误处理和日志记录
    - 请求缓存：缓存常见请求结果
    - 性能监控：跟踪请求性能和成功率
    """

    def __init__(
        self,
        default_timeout: float = 10.0,
        default_max_retries: int = 3,
        enable_cache: bool = True,
        cache_ttl_seconds: int = 300
    ):
        """
        初始化API连接器

        Args:
            default_timeout: 默认超时时间（秒）
            default_max_retries: 默认最大重试次数
            enable_cache: 是否启用缓存
            cache_ttl_seconds: 缓存有效期（秒）
        """
        self.default_timeout = default_timeout
        self.default_max_retries = default_max_retries
        self.enable_cache = enable_cache
        self.cache_ttl_seconds = cache_ttl_seconds

        self.logger = logging.getLogger("APIConnector")

        # 创建会话
        self.session = requests.Session()

        # 配置重试策略
        retry_strategy = Retry(
            total=default_max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # 设置默认headers
        self.session.headers.update({
            'User-Agent': 'Dr.Zero-Agent/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

        # 请求统计
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        # 请求历史
        self.request_history: List[RequestLog] = []
        self.max_history_size = 100

        # 速率限制跟踪
        self.rate_limit_tracker: Dict[str, List[float]] = {}

        # 响应缓存
        self.response_cache: Dict[str, Dict[str, Any]] = {}

        # 注册的API配置
        self.registered_apis: Dict[str, APIConfig] = {}

        self.logger.info(
            f"✅ API连接器初始化完成 - "
            f"超时: {default_timeout}s, 重试: {default_max_retries}次"
        )

    def register_api(
        self,
        name: str,
        base_url: str,
        auth_type: AuthType = AuthType.NONE,
        **auth_params
    ):
        """
        注册API配置

        Args:
            name: API名称
            base_url: 基础URL
            auth_type: 认证类型
            **auth_params: 认证参数
        """
        config = APIConfig(
            base_url=base_url,
            auth_type=auth_type,
            **auth_params
        )

        self.registered_apis[name] = config
        self.logger.info(f"📝 已注册API: {name} ({base_url})")

    def request(
        self,
        url: str,
        method: HttpMethod = HttpMethod.GET,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        api_name: Optional[str] = None,
        timeout: Optional[float] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        发送API请求

        Args:
            url: 请求URL
            method: HTTP方法
            params: URL参数
            data: 请求体数据
            headers: 自定义headers
            api_name: 注册的API名称
            timeout: 超时时间
            use_cache: 是否使用缓存

        Returns:
            API响应字典
        """
        self.total_requests += 1
        start_time = time.time()

        try:
            # 获取API配置
            api_config = self.registered_apis.get(api_name) if api_name else None

            # 构建完整URL
            if api_config:
                full_url = f"{api_config.base_url.rstrip('/')}/{url.lstrip('/')}"

                # 检查速率限制
                self._check_rate_limit(api_name or url, api_config.rate_limit_requests_per_minute)
            else:
                full_url = url

            # 检查缓存
            cache_key = self._generate_cache_key(method.value, full_url, params, data)
            if use_cache and self.enable_cache and cache_key in self.response_cache:
                cached = self.response_cache[cache_key]
                cache_age = time.time() - cached['timestamp']

                if cache_age < self.cache_ttl_seconds:
                    self.logger.debug(f"💾 缓存命中: {method.value} {full_url}")
                    return cached['response']

            # 准备请求
            request_headers = self.session.headers.copy()
            if headers:
                request_headers.update(headers)

            # 添加认证
            if api_config:
                request_headers = self._add_authentication(request_headers, api_config)

            # 发送请求
            timeout = timeout or (api_config.timeout_seconds if api_config else self.default_timeout)

            response = self.session.request(
                method=method.value,
                url=full_url,
                params=params,
                json=data if data else None,
                headers=request_headers,
                timeout=timeout
            )

            # 计算响应时间
            response_time_ms = (time.time() - start_time) * 1000

            # 解析响应
            try:
                response_data = response.json()
            except:
                response_data = response.text

            # 判断成功
            success = response.status_code >= 200 and response.status_code < 300

            # 创建响应对象
            api_response = APIResponse(
                success=success,
                status_code=response.status_code,
                data=response_data,
                headers=dict(response.headers),
                response_time_ms=response_time_ms,
                error_message=None if success else f"HTTP {response.status_code}"
            )

            # 更新统计
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1

            # 记录日志
            self._log_request(method.value, full_url, response.status_code,
                            response_time_ms, success, api_response.error_message)

            # 缓存响应
            if self.enable_cache and success and method == HttpMethod.GET:
                self.response_cache[cache_key] = {
                    'response': api_response.to_dict(),
                    'timestamp': time.time()
                }

                # 限制缓存大小
                if len(self.response_cache) > 100:
                    oldest_key = min(self.response_cache.keys(),
                                   key=lambda k: self.response_cache[k]['timestamp'])
                    del self.response_cache[oldest_key]

            self.logger.info(
                f"{'✅' if success else '❌'} API请求完成 - "
                f"{method.value} {full_url}, "
                f"状态: {response.status_code}, "
                f"耗时: {response_time_ms:.0f}ms"
            )

            return api_response.to_dict()

        except requests.exceptions.Timeout:
            self.failed_requests += 1
            response_time_ms = (time.time() - start_time) * 1000

            self.logger.error(f"⏱️ 请求超时: {method.value} {url}")

            error_response = APIResponse(
                success=False,
                status_code=0,
                data=None,
                response_time_ms=response_time_ms,
                error_message=f"请求超时（{timeout or self.default_timeout}秒）"
            )

            self._log_request(method.value, url, 0, response_time_ms, False,
                            error_response.error_message)

            return error_response.to_dict()

        except requests.exceptions.ConnectionError as e:
            self.failed_requests += 1
            response_time_ms = (time.time() - start_time) * 1000

            self.logger.error(f"🔌 连接错误: {str(e)}")

            error_response = APIResponse(
                success=False,
                status_code=0,
                data=None,
                response_time_ms=response_time_ms,
                error_message=f"连接错误: {str(e)}"
            )

            self._log_request(method.value, url, 0, response_time_ms, False,
                            error_response.error_message)

            return error_response.to_dict()

        except Exception as e:
            self.failed_requests += 1
            response_time_ms = (time.time() - start_time) * 1000

            self.logger.error(f"❌ 请求失败: {str(e)}")

            error_response = APIResponse(
                success=False,
                status_code=0,
                data=None,
                response_time_ms=response_time_ms,
                error_message=str(e)
            )

            self._log_request(method.value, url, 0, response_time_ms, False,
                            error_response.error_message)

            return error_response.to_dict()

    def get(self, url: str, **kwargs) -> Dict[str, Any]:
        """GET请求"""
        return self.request(url, method=HttpMethod.GET, **kwargs)

    def post(self, url: str, data: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """POST请求"""
        return self.request(url, method=HttpMethod.POST, data=data, **kwargs)

    def put(self, url: str, data: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """PUT请求"""
        return self.request(url, method=HttpMethod.PUT, data=data, **kwargs)

    def delete(self, url: str, **kwargs) -> Dict[str, Any]:
        """DELETE请求"""
        return self.request(url, method=HttpMethod.DELETE, **kwargs)

    def _add_authentication(
        self,
        headers: Dict[str, str],
        config: APIConfig
    ) -> Dict[str, str]:
        """添加认证信息"""
        if config.auth_type == AuthType.API_KEY and config.api_key:
            headers['Authorization'] = f"Api-Key {config.api_key}"

        elif config.auth_type == AuthType.BEARER_TOKEN and config.bearer_token:
            headers['Authorization'] = f"Bearer {config.bearer_token}"

        elif config.auth_type == AuthType.BASIC_AUTH:
            if config.username and config.password:
                import base64
                credentials = base64.b64encode(
                    f"{config.username}:{config.password}".encode()
                ).decode()
                headers['Authorization'] = f"Basic {credentials}"

        return headers

    def _check_rate_limit(self, key: str, max_rpm: int):
        """检查速率限制"""
        now = time.time()

        if key not in self.rate_limit_tracker:
            self.rate_limit_tracker[key] = []

        # 清理旧记录（超过1分钟的）
        self.rate_limit_tracker[key] = [
            t for t in self.rate_limit_tracker[key]
            if now - t < 60
        ]

        # 检查是否超过限制
        if len(self.rate_limit_tracker[key]) >= max_rpm:
            wait_time = 60 - (now - self.rate_limit_tracker[key][0])
            if wait_time > 0:
                self.logger.warning(f"⏱️ 速率限制，等待 {wait_time:.1f}秒")
                time.sleep(wait_time)

        # 记录本次请求
        self.rate_limit_tracker[key].append(now)

    def _generate_cache_key(
        self,
        method: str,
        url: str,
        params: Optional[Dict],
        data: Optional[Dict]
    ) -> str:
        """生成缓存键"""
        key_parts = [method, url]

        if params:
            key_parts.append(json.dumps(params, sort_keys=True))

        if data:
            key_parts.append(json.dumps(data, sort_keys=True))

        return '|'.join(key_parts)

    def _log_request(
        self,
        method: str,
        url: str,
        status_code: int,
        response_time_ms: float,
        success: bool,
        error: Optional[str] = None
    ):
        """记录请求日志"""
        log_entry = RequestLog(
            timestamp=datetime.now(),
            method=method,
            url=url,
            status_code=status_code,
            response_time_ms=response_time_ms,
            success=success,
            error=error
        )

        self.request_history.append(log_entry)

        # 限制历史记录大小
        if len(self.request_history) > self.max_history_size:
            self.request_history.pop(0)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(
                self.successful_requests / max(self.total_requests, 1) * 100, 2
            ),
            "registered_apis": list(self.registered_apis.keys()),
            "cache_size": len(self.response_cache),
            "request_history_size": len(self.request_history),
            "recent_requests": [
                log.to_dict() for log in self.request_history[-10:]
            ]
        }

    def clear_cache(self):
        """清空缓存"""
        self.response_cache.clear()
        self.logger.info("🧹 API响应缓存已清空")

    def clear_history(self):
        """清空历史记录"""
        self.request_history.clear()
        self.logger.info("🧹 请求历史已清空")

    def __del__(self):
        """析构函数，清理资源"""
        self.session.close()
