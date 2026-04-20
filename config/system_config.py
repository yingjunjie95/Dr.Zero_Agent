class SystemConfig:
    """4 vCPU, 16GB内存环境下的优化配置"""

    # ==================== 硬件配置 ====================
    CPU_CORES = 4
    MEMORY_GB = 16
    MAX_MEMORY_USAGE = 0.85  # 最大内存使用率85%

    # ==================== 模型配置 ====================
    MODEL_NAME = "Qwen/Qwen1.5-4B-Chat"
    QUANTIZATION = "4bit"  # 4-bit量化
    MAX_CONTEXT_LENGTH = 2048
    TEMPERATURE = 0.7  # 生成温度，控制随机性
    TOP_P = 0.9  # 核采样参数
    MAX_NEW_TOKENS = 512  # 最大生成token数

    # ==================== 性能配置 ====================
    MAX_RESPONSE_TIME = 5.0  # 秒
    BATCH_SIZE = 2  # CPU优化的batch size
    THREAD_COUNT = 4  # 匹配CPU核心数
    CACHE_ENABLED = True  # 启用缓存
    CACHE_MAX_SIZE = 1000  # 缓存最大条目数

    # ==================== 安全配置 ====================
    TOOL_EXECUTION_TIMEOUT = 30  # 秒
    MAX_TOOL_CALLS_PER_SESSION = 10
    RISK_THRESHOLD = 0.7  # 风险阈值
    ENABLE_INPUT_VALIDATION = True  # 启用输入验证
    MAX_INPUT_LENGTH = 4096  # 最大输入长度

    # ==================== 日志配置 ====================
    LOG_LEVEL = "INFO"  # 日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = "logs/system.log"
    LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5  # 日志文件备份数量

    # ==================== 会话配置 ====================
    SESSION_TIMEOUT = 1800  # 会话超时时间（秒），默认30分钟
    MAX_SESSION_HISTORY = 50  # 最大会话历史消息数
    ENABLE_CONVERSATION_COMPRESSION = True  # 启用对话压缩

    # ==================== 存储配置 ====================
    DATA_DIR = "data"
    MODEL_CACHE_DIR = "cache/models"
    VECTOR_DB_PATH = "data/vector_db"
    CHECKPOINT_INTERVAL = 100  # 检查点保存间隔（步数）

    # ==================== 网络配置 ====================
    API_TIMEOUT = 30  # API请求超时时间（秒）
    MAX_RETRIES = 3  # 最大重试次数
    RETRY_DELAY = 1  # 重试延迟（秒）

    @classmethod
    def get_available_memory(cls) -> float:
        """获取可用内存（GB）"""
        return cls.MEMORY_GB * cls.MAX_MEMORY_USAGE

    @classmethod
    def is_production(cls) -> bool:
        """判断是否为生产环境"""
        import os
        return os.getenv("ENVIRONMENT", "development").lower() == "production"

    @classmethod
    def get_model_config(cls) -> dict:
        """获取模型配置字典"""
        return {
            "model_name": cls.MODEL_NAME,
            "quantization": cls.QUANTIZATION,
            "max_context_length": cls.MAX_CONTEXT_LENGTH,
            "temperature": cls.TEMPERATURE,
            "top_p": cls.TOP_P,
            "max_new_tokens": cls.MAX_NEW_TOKENS,
        }
