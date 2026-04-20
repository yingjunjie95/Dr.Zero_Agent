"""
环境监控器 - Environment Monitor
负责监控系统资源、外部环境变化和Agent运行状态
为Agent的自主决策提供实时环境信息
"""
import psutil
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class ResourceLevel(Enum):
    """资源使用等级"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SystemMetrics:
    """系统指标数据类"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_percent: float
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    load_average: float = 0.0

    def get_resource_level(self) -> Dict[str, ResourceLevel]:
        """获取各资源的 usage 等级"""
        return {
            "cpu": self._classify_resource(self.cpu_percent, [60, 80, 95]),
            "memory": self._classify_resource(self.memory_percent, [70, 85, 95]),
            "disk": self._classify_resource(self.disk_percent, [75, 90, 98]),
        }

    @staticmethod
    def _classify_resource(value: float, thresholds: List[float]) -> ResourceLevel:
        """根据阈值分类资源使用等级"""
        if value < thresholds[0]:
            return ResourceLevel.LOW
        elif value < thresholds[1]:
            return ResourceLevel.NORMAL
        elif value < thresholds[2]:
            return ResourceLevel.HIGH
        else:
            return ResourceLevel.CRITICAL


@dataclass
class EnvironmentState:
    """环境状态快照"""
    system_metrics: SystemMetrics
    active_tasks: int
    queue_size: int
    error_rate: float
    response_time_avg: float
    last_update: datetime

    def is_resource_constrained(self) -> bool:
        """判断是否处于资源受限状态"""
        levels = self.system_metrics.get_resource_level()
        return any(level in [ResourceLevel.HIGH, ResourceLevel.CRITICAL]
                  for level in levels.values())


class EnvironmentMonitor:
    """
    环境监控器

    功能：
    - 实时监控系统资源（CPU、内存、磁盘、网络）
    - 检测环境变化和异常
    - 提供资源使用趋势分析
    - 触发资源预警和优化建议
    """

    def __init__(self, monitoring_interval: float = 2.0, history_size: int = 100):
        """
        初始化环境监控器

        Args:
            monitoring_interval: 监控采样间隔（秒）
            history_size: 历史数据保留数量
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size

        # 日志配置
        self.logger = logging.getLogger("EnvironmentMonitor")

        # 监控数据存储
        self.metrics_history: deque = deque(maxlen=history_size)
        self.current_metrics: Optional[SystemMetrics] = None
        self.environment_state: Optional[EnvironmentState] = None

        # 任务跟踪
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks_count: int = 0
        self.error_count: int = 0
        self.total_requests: int = 0

        # 响应时间跟踪
        self.response_times: deque = deque(maxlen=50)

        # 预警回调
        self.warning_callbacks: List[callable] = []

        # 监控状态
        self.is_monitoring: bool = False
        self.monitor_thread: Optional[threading.Thread] = None

        # 性能基线（用于异常检测）
        self.baseline_cpu: float = 0.0
        self.baseline_memory: float = 0.0
        self.baseline_initialized: bool = False

        self.logger.info("环境监控器初始化完成")

    def start_monitoring(self):
        """启动后台监控线程"""
        if self.is_monitoring:
            self.logger.warning("监控已在运行中")
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="EnvironmentMonitor"
        )
        self.monitor_thread.start()
        self.logger.info("✅ 环境监控已启动")

    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("⏹ 环境监控已停止")

    def _monitoring_loop(self):
        """监控主循环"""
        while self.is_monitoring:
            try:
                self._collect_metrics()
                self._update_environment_state()
                self._detect_anomalies()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"监控循环错误: {str(e)}", exc_info=True)
                time.sleep(self.monitoring_interval * 2)

    def _collect_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        try:
            # CPU信息
            cpu_percent = psutil.cpu_percent(interval=0.5)

            # 内存信息
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024 ** 3)
            memory_available_gb = memory.available / (1024 ** 3)

            # 磁盘信息
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            # 网络信息
            net_io = psutil.net_io_counters()
            network_sent_mb = net_io.bytes_sent / (1024 ** 2) if net_io else 0.0
            network_recv_mb = net_io.bytes_recv / (1024 ** 2) if net_io else 0.0

            # 系统负载（仅Unix系统）
            try:
                load_average = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            except:
                load_average = 0.0

            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                memory_available_gb=memory_available_gb,
                disk_percent=disk_percent,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                load_average=load_average
            )

            # 更新当前指标和历史记录
            self.current_metrics = metrics
            self.metrics_history.append(metrics)

            # 初始化基线
            if not self.baseline_initialized and len(self.metrics_history) >= 10:
                self._initialize_baseline()

            return metrics

        except Exception as e:
            self.logger.error(f"收集系统指标失败: {str(e)}")
            raise

    def _initialize_baseline(self):
        """初始化性能基线"""
        recent_metrics = list(self.metrics_history)[-10:]
        self.baseline_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        self.baseline_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        self.baseline_initialized = True
        self.logger.info(f"性能基线已建立 - CPU: {self.baseline_cpu:.1f}%, Memory: {self.baseline_memory:.1f}%")

    def _update_environment_state(self):
        """更新环境状态"""
        if not self.current_metrics:
            return

        # 计算平均响应时间
        avg_response_time = (
            sum(self.response_times) / len(self.response_times)
            if self.response_times else 0.0
        )

        # 计算错误率
        error_rate = (
            self.error_count / self.total_requests
            if self.total_requests > 0 else 0.0
        )

        self.environment_state = EnvironmentState(
            system_metrics=self.current_metrics,
            active_tasks=len(self.active_tasks),
            queue_size=len(self.active_tasks),
            error_rate=error_rate,
            response_time_avg=avg_response_time,
            last_update=datetime.now()
        )

    def _detect_anomalies(self):
        """检测异常情况并触发预警"""
        if not self.current_metrics or not self.baseline_initialized:
            return

        warnings = []

        # CPU异常检测
        if self.current_metrics.cpu_percent > self.baseline_cpu * 2.0:
            warnings.append({
                "type": "cpu_spike",
                "severity": "warning",
                "message": f"CPU使用率异常升高: {self.current_metrics.cpu_percent:.1f}%",
                "value": self.current_metrics.cpu_percent
            })

        # 内存异常检测
        if self.current_metrics.memory_percent > self.baseline_memory * 1.5:
            warnings.append({
                "type": "memory_leak",
                "severity": "warning",
                "message": f"内存使用率异常: {self.current_metrics.memory_percent:.1f}%",
                "value": self.current_metrics.memory_percent
            })

        # 高负载检测
        resource_levels = self.current_metrics.get_resource_level()
        for resource, level in resource_levels.items():
            if level == ResourceLevel.CRITICAL:
                warnings.append({
                    "type": f"{resource}_critical",
                    "severity": "critical",
                    "message": f"{resource.upper()}资源达到临界水平",
                    "value": getattr(self.current_metrics, f"{resource}_percent", 0)
                })

        # 错误率过高
        if self.total_requests > 10:
            error_rate = self.error_count / self.total_requests
            if error_rate > 0.1:  # 错误率超过10%
                warnings.append({
                    "type": "high_error_rate",
                    "severity": "warning",
                    "message": f"错误率过高: {error_rate:.2%}",
                    "value": error_rate
                })

        # 触发预警回调
        for warning in warnings:
            self.logger.warning(f"⚠️ {warning['message']}")
            for callback in self.warning_callbacks:
                try:
                    callback(warning)
                except Exception as e:
                    self.logger.error(f"预警回调执行失败: {str(e)}")

    def register_task(self, task_id: str, task_info: Dict[str, Any]):
        """注册新任务"""
        self.active_tasks[task_id] = {
            "info": task_info,
            "start_time": time.time(),
            "status": "running"
        }
        self.logger.debug(f"任务注册: {task_id}")

    def complete_task(self, task_id: str, success: bool = True, response_time: float = None):
        """完成任务"""
        if task_id in self.active_tasks:
            task_info = self.active_tasks.pop(task_id)
            self.completed_tasks_count += 1
            self.total_requests += 1

            if not success:
                self.error_count += 1

            if response_time:
                self.response_times.append(response_time)

            elapsed = time.time() - task_info["start_time"]
            self.logger.debug(f"任务完成: {task_id}, 耗时: {elapsed:.2f}s, 成功: {success}")

    def get_current_state(self) -> Optional[EnvironmentState]:
        """获取当前环境状态"""
        return self.environment_state

    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        if not self.current_metrics:
            return {}

        return {
            "current": {
                "cpu_percent": self.current_metrics.cpu_percent,
                "memory_percent": self.current_metrics.memory_percent,
                "memory_used_gb": round(self.current_metrics.memory_used_gb, 2),
                "memory_available_gb": round(self.current_metrics.memory_available_gb, 2),
                "disk_percent": self.current_metrics.disk_percent,
            },
            "tasks": {
                "active": len(self.active_tasks),
                "completed": self.completed_tasks_count,
                "error_rate": round(self.error_count / max(self.total_requests, 1), 3),
            },
            "performance": {
                "avg_response_time": round(
                    sum(self.response_times) / max(len(self.response_times), 1), 3
                ),
                "baseline_cpu": round(self.baseline_cpu, 1),
                "baseline_memory": round(self.baseline_memory, 1),
            }
        }

    def get_resource_trend(self, metric: str, window: int = 20) -> List[float]:
        """
        获取资源使用趋势

        Args:
            metric: 指标名称 ('cpu', 'memory', 'disk')
            window: 时间窗口大小

        Returns:
            指标值列表
        """
        if len(self.metrics_history) < window:
            return []

        recent_metrics = list(self.metrics_history)[-window:]
        attribute_map = {
            "cpu": "cpu_percent",
            "memory": "memory_percent",
            "disk": "disk_percent"
        }

        attr = attribute_map.get(metric)
        if not attr:
            raise ValueError(f"未知指标: {metric}")

        return [getattr(m, attr) for m in recent_metrics]

    def should_throttle_operations(self) -> bool:
        """判断是否应该限制操作以节省资源"""
        if not self.current_metrics:
            return False

        # 当CPU或内存处于HIGH或CRITICAL时，建议限制操作
        levels = self.current_metrics.get_resource_level()
        return any(
            level in [ResourceLevel.HIGH, ResourceLevel.CRITICAL]
            for level in [levels["cpu"], levels["memory"]]
        )

    def get_optimization_suggestions(self) -> List[str]:
        """获取优化建议"""
        suggestions = []

        if not self.current_metrics:
            return suggestions

        levels = self.current_metrics.get_resource_level()

        # CPU优化建议
        if levels["cpu"] == ResourceLevel.CRITICAL:
            suggestions.append("CPU使用率过高，建议减少并发任务或优化计算密集型操作")
        elif levels["cpu"] == ResourceLevel.HIGH:
            suggestions.append("CPU使用率较高，考虑批量处理任务以提高效率")

        # 内存优化建议
        if levels["memory"] == ResourceLevel.CRITICAL:
            suggestions.append("内存使用率临界，建议清理缓存或释放未使用的记忆数据")
        elif levels["memory"] == ResourceLevel.HIGH:
            suggestions.append("内存使用率较高，可以考虑压缩对话历史或归档旧记忆")

        # 错误率建议
        if self.total_requests > 20:
            error_rate = self.error_count / self.total_requests
            if error_rate > 0.05:
                suggestions.append(f"错误率偏高({error_rate:.1%})，建议检查工具调用和输入验证")

        # 响应时间建议
        if self.response_times:
            avg_time = sum(self.response_times) / len(self.response_times)
            if avg_time > 5.0:
                suggestions.append(f"平均响应时间较长({avg_time:.1f}s)，考虑启用缓存或简化处理流程")

        return suggestions

    def add_warning_callback(self, callback: callable):
        """添加预警回调函数"""
        self.warning_callbacks.append(callback)

    def reset_statistics(self):
        """重置统计数据"""
        self.active_tasks.clear()
        self.completed_tasks_count = 0
        self.error_count = 0
        self.total_requests = 0
        self.response_times.clear()
        self.logger.info("统计数据已重置")

    def __enter__(self):
        """上下文管理器入口"""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_monitoring()
        return False
