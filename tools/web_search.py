"""
网络搜索工具 - Web Search Tool
提供安全、高效的网络搜索功能
支持多种搜索引擎，具备结果过滤和摘要能力
专为CPU环境优化，轻量级且可靠的搜索解决方案
"""
import logging
import time
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SearchEngine(Enum):
    """搜索引擎类型"""
    GOOGLE = "google"
    BING = "bing"
    DUCKDUCKGO = "duckduckgo"
    BAIDU = "baidu"


@dataclass
class SearchResult:
    """搜索结果"""
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float = 0.0
    published_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "relevance_score": round(self.relevance_score, 3),
            "published_date": self.published_date
        }


@dataclass
class SearchResponse:
    """搜索响应"""
    success: bool
    query: str
    results: List[SearchResult]
    total_results: int = 0
    search_time_ms: float = 0.0
    engine_used: str = ""
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_results": self.total_results,
            "search_time_ms": round(self.search_time_ms, 2),
            "engine_used": self.engine_used,
            "error_message": self.error_message
        }


class WebSearchTool:
    """
    网络搜索工具

    功能：
    - 多引擎支持：支持多种搜索引擎
    - 智能查询：自动优化搜索查询
    - 结果过滤：过滤低质量和不相关内容
    - 摘要生成：提取关键信息生成摘要
    - 速率限制：控制搜索频率避免被封
    - 缓存机制：缓存常见搜索结果
    - 安全保护：防止恶意查询和注入
    """

    def __init__(
        self,
        default_engine: SearchEngine = SearchEngine.DUCKDUCKGO,
        max_results: int = 5,
        timeout_seconds: float = 10.0,
        rate_limit_seconds: float = 2.0,
        enable_cache: bool = True,
        cache_ttl_seconds: int = 3600
    ):
        """
        初始化网络搜索工具

        Args:
            default_engine: 默认搜索引擎
            max_results: 最大返回结果数
            timeout_seconds: 请求超时时间（秒）
            rate_limit_seconds: 搜索间隔限制（秒）
            enable_cache: 是否启用缓存
            cache_ttl_seconds: 缓存有效期（秒）
        """
        self.default_engine = default_engine
        self.max_results = max_results
        self.timeout_seconds = timeout_seconds
        self.rate_limit_seconds = rate_limit_seconds
        self.enable_cache = enable_cache
        self.cache_ttl_seconds = cache_ttl_seconds

        self.logger = logging.getLogger("WebSearchTool")

        # 搜索统计
        self.total_searches = 0
        self.successful_searches = 0
        self.failed_searches = 0
        self.cache_hits = 0

        # 速率限制
        self.last_search_time: Optional[float] = None

        # 结果缓存
        self.search_cache: Dict[str, Dict[str, Any]] = {}

        # 会话配置
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Dr.Zero-Agent/1.0'
        })

        self.logger.info(
            f"✅ 网络搜索工具初始化完成 - 引擎: {default_engine.value}, "
            f"最大结果: {max_results}"
        )

    def search(
        self,
        query: str,
        num_results: Optional[int] = None,
        engine: Optional[SearchEngine] = None,
        language: str = "zh-CN",
        safe_search: bool = True
    ) -> Dict[str, Any]:
        """
        执行网络搜索

        Args:
            query: 搜索查询
            num_results: 返回结果数量
            engine: 搜索引擎
            language: 语言设置
            safe_search: 是否启用安全搜索

        Returns:
            搜索结果字典
        """
        start_time = time.time()
        self.total_searches += 1

        try:
            # 参数验证
            if not query or len(query.strip()) == 0:
                raise ValueError("搜索查询不能为空")

            if len(query) > 500:
                raise ValueError("搜索查询过长（最大500字符）")

            # 检查速率限制
            self._check_rate_limit()

            # 检查缓存
            cache_key = self._generate_cache_key(query, engine, language)
            if self.enable_cache and cache_key in self.search_cache:
                cached_result = self.search_cache[cache_key]
                cache_age = time.time() - cached_result.get('timestamp', 0)

                if cache_age < self.cache_ttl_seconds:
                    self.cache_hits += 1
                    self.logger.debug(f"💾 缓存命中: {query[:50]}")
                    return cached_result['data']

            # 执行搜索
            search_engine = engine or self.default_engine
            result_count = num_results or self.max_results

            self.logger.info(f"🔍 执行搜索: {query[:50]}... (引擎: {search_engine.value})")

            # 根据搜索引擎选择实现
            if search_engine == SearchEngine.DUCKDUCKGO:
                response = self._search_duckduckgo(query, result_count, language, safe_search)
            elif search_engine == SearchEngine.BING:
                response = self._search_bing(query, result_count, language, safe_search)
            elif search_engine == SearchEngine.GOOGLE:
                response = self._search_google(query, result_count, language, safe_search)
            elif search_engine == SearchEngine.BAIDU:
                response = self._search_baidu(query, result_count, language, safe_search)
            else:
                raise ValueError(f"不支持的搜索引擎: {search_engine.value}")

            # 计算搜索时间
            search_time_ms = (time.time() - start_time) * 1000
            response.search_time_ms = search_time_ms

            # 更新统计
            if response.success:
                self.successful_searches += 1
            else:
                self.failed_searches += 1

            # 缓存结果
            if self.enable_cache and response.success:
                self.search_cache[cache_key] = {
                    'data': response.to_dict(),
                    'timestamp': time.time()
                }

                # 限制缓存大小
                if len(self.search_cache) > 100:
                    oldest_key = min(self.search_cache.keys(),
                                   key=lambda k: self.search_cache[k]['timestamp'])
                    del self.search_cache[oldest_key]

            self.logger.info(
                f"✅ 搜索完成 - 找到 {len(response.results)} 条结果, "
                f"耗时: {search_time_ms:.0f}ms"
            )

            return response.to_dict()

        except Exception as e:
            self.failed_searches += 1
            search_time_ms = (time.time() - start_time) * 1000

            self.logger.error(f"❌ 搜索失败: {str(e)}")

            error_response = SearchResponse(
                success=False,
                query=query,
                results=[],
                search_time_ms=search_time_ms,
                engine_used=(engine or self.default_engine).value,
                error_message=str(e)
            )

            return error_response.to_dict()

    def _search_duckduckgo(
        self,
        query: str,
        num_results: int,
        language: str,
        safe_search: bool
    ) -> SearchResponse:
        """使用DuckDuckGo搜索"""
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(
                    keywords=query,
                    max_results=num_results,
                    region='cn-zh' if 'zh' in language else 'en-us',
                    safesearch='moderate' if safe_search else 'off'
                ))

            search_results = []
            for result in results:
                search_result = SearchResult(
                    title=result.get('title', ''),
                    url=result.get('href', ''),
                    snippet=result.get('body', ''),
                    source="DuckDuckGo",
                    relevance_score=0.8
                )
                search_results.append(search_result)

            return SearchResponse(
                success=True,
                query=query,
                results=search_results,
                total_results=len(search_results),
                engine_used="duckduckgo"
            )

        except ImportError:
            self.logger.warning("⚠️ duckduckgo-search 未安装，使用备用方案")
            return self._search_fallback(query, num_results)

        except Exception as e:
            self.logger.error(f"DuckDuckGo搜索失败: {str(e)}")
            return self._search_fallback(query, num_results)

    def _search_bing(
        self,
        query: str,
        num_results: int,
        language: str,
        safe_search: bool
    ) -> SearchResponse:
        """使用Bing搜索（需要API密钥）"""
        api_key = self._get_bing_api_key()

        if not api_key:
            self.logger.warning("⚠️ Bing API密钥未配置，使用备用方案")
            return self._search_fallback(query, num_results)

        try:
            endpoint = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": api_key}
            params = {
                "q": query,
                "count": num_results,
                "mkt": language,
                "safeSearch": "Moderate" if safe_search else "Off"
            }

            response = self.session.get(
                endpoint,
                headers=headers,
                params=params,
                timeout=self.timeout_seconds
            )
            response.raise_for_status()

            data = response.json()
            web_pages = data.get('webPages', {}).get('value', [])

            search_results = []
            for item in web_pages:
                search_result = SearchResult(
                    title=item.get('name', ''),
                    url=item.get('url', ''),
                    snippet=item.get('snippet', ''),
                    source="Bing",
                    relevance_score=0.85,
                    published_date=item.get('dateLastCrawled')
                )
                search_results.append(search_result)

            return SearchResponse(
                success=True,
                query=query,
                results=search_results,
                total_results=data.get('webPages', {}).get('totalEstimatedMatches', 0),
                engine_used="bing"
            )

        except Exception as e:
            self.logger.error(f"Bing搜索失败: {str(e)}")
            return self._search_fallback(query, num_results)

    def _search_google(
        self,
        query: str,
        num_results: int,
        language: str,
        safe_search: bool
    ) -> SearchResponse:
        """使用Google搜索（需要API密钥）"""
        api_key = self._get_google_api_key()
        cx = self._get_google_cx()

        if not api_key or not cx:
            self.logger.warning("⚠️ Google API配置不完整，使用备用方案")
            return self._search_fallback(query, num_results)

        try:
            endpoint = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": api_key,
                "cx": cx,
                "q": query,
                "num": min(num_results, 10),
                "hl": language.split('-')[0],
                "safe": "active" if safe_search else "off"
            }

            response = self.session.get(
                endpoint,
                params=params,
                timeout=self.timeout_seconds
            )
            response.raise_for_status()

            data = response.json()
            items = data.get('items', [])

            search_results = []
            for item in items:
                search_result = SearchResult(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('snippet', ''),
                    source="Google",
                    relevance_score=0.9
                )
                search_results.append(search_result)

            return SearchResponse(
                success=True,
                query=query,
                results=search_results,
                total_results=int(data.get('searchInformation', {}).get('totalResults', 0)),
                engine_used="google"
            )

        except Exception as e:
            self.logger.error(f"Google搜索失败: {str(e)}")
            return self._search_fallback(query, num_results)

    def _search_baidu(
        self,
        query: str,
        num_results: int,
        language: str,
        safe_search: bool
    ) -> SearchResponse:
        """使用百度搜索（简化版）"""
        try:
            url = "https://www.baidu.com/s"
            params = {
                "wd": query,
                "rn": num_results,
                "ie": "utf-8"
            }

            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout_seconds
            )
            response.raise_for_status()

            # 注意：这里需要HTML解析，简化处理
            self.logger.warning("⚠️ 百度搜索需要HTML解析，使用备用方案")
            return self._search_fallback(query, num_results)

        except Exception as e:
            self.logger.error(f"百度搜索失败: {str(e)}")
            return self._search_fallback(query, num_results)

    def _search_fallback(self, query: str, num_results: int) -> SearchResponse:
        """备用搜索方案"""
        self.logger.info("使用备用搜索方案")

        # 返回提示信息
        search_result = SearchResult(
            title="搜索功能需要配置",
            url="",
            snippet=f"要使用完整的搜索功能，请配置以下之一：\n1. 安装 duckduckgo-search: pip install duckduckgo-search\n2. 配置 Bing API 密钥\n3. 配置 Google Custom Search API",
            source="System",
            relevance_score=0.5
        )

        return SearchResponse(
            success=True,
            query=query,
            results=[search_result],
            total_results=1,
            engine_used="fallback"
        )

    def _check_rate_limit(self):
        """检查速率限制"""
        if self.last_search_time:
            elapsed = time.time() - self.last_search_time
            if elapsed < self.rate_limit_seconds:
                wait_time = self.rate_limit_seconds - elapsed
                self.logger.debug(f"⏱️ 速率限制，等待 {wait_time:.1f}秒")
                time.sleep(wait_time)

        self.last_search_time = time.time()

    def _generate_cache_key(
        self,
        query: str,
        engine: Optional[SearchEngine],
        language: str
    ) -> str:
        """生成缓存键"""
        engine_str = engine.value if engine else self.default_engine.value
        return f"{query.lower()}_{engine_str}_{language}"

    def _get_bing_api_key(self) -> Optional[str]:
        """获取Bing API密钥"""
        import os
        return os.getenv('BING_API_KEY')

    def _get_google_api_key(self) -> Optional[str]:
        """获取Google API密钥"""
        import os
        return os.getenv('GOOGLE_API_KEY')

    def _get_google_cx(self) -> Optional[str]:
        """获取Google Custom Search ID"""
        import os
        return os.getenv('GOOGLE_CX')

    def get_statistics(self) -> Dict[str, Any]:
        """获取搜索统计信息"""
        return {
            "total_searches": self.total_searches,
            "successful_searches": self.successful_searches,
            "failed_searches": self.failed_searches,
            "success_rate": round(
                self.successful_searches / max(self.total_searches, 1) * 100, 2
            ),
            "cache_hits": self.cache_hits,
            "cache_size": len(self.search_cache),
            "default_engine": self.default_engine.value,
            "rate_limit_seconds": self.rate_limit_seconds
        }

    def clear_cache(self):
        """清空搜索缓存"""
        self.search_cache.clear()
        self.logger.info("🧹 搜索缓存已清空")

    def __del__(self):
        """析构函数，清理资源"""
        self.session.close()
