"""
性能跟踪器 - Performance Tracker
实时监控和记录Agent的性能指标
提供详细的性能分析、趋势追踪和资源使用监控
专为CPU环境优化，轻量级且高效的性能监控解决方案
"""
import logging
import time
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict


@dataclass
class InteractionRecord:
    """交互记录"""
    timestamp: datetime
    processing_time: float
    success: bool
    execution_metrics: Dict[str, Any]
    cognitive_state: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "processing_time": round(self.processing_time, 3),
            "success": self.success,
            "execution_metrics": self.execution_metrics,
            "cognitive_state": {
                "confidence_score": self.cognitive_state.get("confidence_score", 0.0),
                "risk_level": self.cognitive_state.get("risk_level", 0.0)
            }
        }


@dataclass
class PerformanceMetrics:
    """性能指标"""
    # 基础统计
    total_interactions: int = 0
    successful_interactions: int = 0
    failed_interactions: int = 0

    # 时间统计
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    max_processing_time: float = 0.0

    # 成功率
    success_rate: float = 0.0

    # 工具使用统计
    total_tool_calls: int = 0
    average_tool_calls_per_interaction: float = 0.0

    # 错误统计
    total_errors: int = 0
    error_rate: float = 0.0

    # 置信度统计
    average_confidence: float = 0.0
    total_confidence_sum: float = 0.0

    def update(self, processing_time: float, execution_metrics: Dict[str, Any],
               cognitive_state: Dict[str, Any], success: bool):
        """更新性能指标"""
        self.total_interactions += 1

        if success:
            self.successful_interactions += 1
        else:
            self.failed_interactions += 1

        # 更新时间统计
        self.total_processing_time += processing_time
        self.average_processing_time = self.total_processing_time / self.total_interactions
        self.min_processing_time = min(self.min_processing_time, processing_time)
        self.max_processing_time = max(self.max_processing_time, processing_time)

        # 更新成功率
        self.success_rate = self.successful_interactions / self.total_interactions

        # 更新工具使用统计
        tool_calls = execution_metrics.get('tool_calls', 0)
        self.total_tool_calls += tool_calls
        self.average_tool_calls_per_interaction = (
            self.total_tool_calls / self.total_interactions
        )

        # 更新错误统计
        errors = execution_metrics.get('errors', 0)
        self.total_errors += errors
        self.error_rate = self.total_errors / self.total_interactions if self.total_interactions > 0 else 0.0

        # 更新置信度统计
        confidence = cognitive_state.get('confidence_score', 0.0)
        self.total_confidence_sum += confidence
        self.average_confidence = self.total_confidence_sum / self.total_interactions

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_interactions": self.total_interactions,
            "successful_interactions": self.successful_interactions,
            "failed_interactions": self.failed_interactions,
            "success_rate": round(self.success_rate, 3),
            "average_processing_time": round(self.average_processing_time, 3),
            "min_processing_time": round(self.min_processing_time, 3) if self.min_processing_time != float('inf') else 0.0,
            "max_processing_time": round(self.max_processing_time, 3),
            "total_tool_calls": self.total_tool_calls,
            "average_tool_calls_per_interaction": round(self.average_tool_calls_per_interaction, 3),
            "total_errors": self.total_errors,
            "error_rate": round(self.error_rate, 3),
            "average_confidence": round(self.average_confidence, 3)
        }


class PerformanceTracker:
    """
    性能跟踪器

    功能：
    - 交互记录：记录每次交互的性能数据
    - 实时统计：计算和维护各种性能指标
    - 趋势分析：分析性能变化趋势
    - 资源监控：监控系统资源使用情况
    - 告警机制：在性能异常时发出告警
    - 数据导出：提供性能数据导出功能
    """

    def __init__(
        self,
        window_size: int = 100,
        alert_threshold_response_time: float = 5.0,
        alert_threshold_error_rate: float = 0.3
    ):
        """
        初始化性能跟踪器

        Args:
            window_size: 滑动窗口大小（用于近期性能计算）
            alert_threshold_response_time: 响应时间告警阈值（秒）
            alert_threshold_error_rate: 错误率告警阈值
        """
        self.window_size = window_size
        self.alert_threshold_response_time = alert_threshold_response_time
        self.alert_threshold_error_rate = alert_threshold_error_rate

        self.logger = logging.getLogger("PerformanceTracker")

        # 性能指标
        self.metrics = PerformanceMetrics()

        # 交互历史
        self.interaction_history: deque = deque(maxlen=1000)

        # 近期性能（滑动窗口）
        self.recent_performance: deque = deque(maxlen=window_size)

        # 时间序列数据
        self.time_series_data: List[Dict[str, Any]] = []

        # 告警历史
        self.alerts: List[Dict[str, Any]] = []

        # 监控状态
        self.is_monitoring = False
        self.monitoring_start_time: Optional[datetime] = None

        # 资源使用历史
        self.resource_usage_history: deque = deque(maxlen=100)

        # 后台监控线程
        self.monitoring_thread: Optional[threading.Thread] = None

        self.logger.info(
            f"✅ 性能跟踪器初始化完成 - 窗口大小: {window_size}"
        )

    def record_interaction(
        self,
        processing_time: float,
        execution_metrics: Dict[str, Any],
        cognitive_state: Dict[str, Any]
    ):
        """
        记录交互性能数据

        Args:
            processing_time: 处理时间（秒）
            execution_metrics: 执行指标
            cognitive_state: 认知状态
        """
        try:
            # 判断是否成功
            success = execution_metrics.get('errors', 0) == 0

            # 更新总体指标
            self.metrics.update(processing_time, execution_metrics, cognitive_state, success)

            # 创建交互记录
            record = InteractionRecord(
                timestamp=datetime.now(),
                processing_time=processing_time,
                success=success,
                execution_metrics=execution_metrics,
                cognitive_state=cognitive_state
            )

            # 保存到历史
            self.interaction_history.append(record)
            self.recent_performance.append(record)

            # 记录时间序列数据
            self.time_series_data.append({
                "timestamp": record.timestamp.isoformat(),
                "processing_time": processing_time,
                "success": success,
                "tool_calls": execution_metrics.get('tool_calls', 0),
                "errors": execution_metrics.get('errors', 0),
                "confidence": cognitive_state.get('confidence_score', 0.0)
            })

            # 限制时间序列数据大小
            if len(self.time_series_data) > 500:
                self.time_series_data.pop(0)

            # 检查是否需要告警
            self._check_alerts(processing_time, success, execution_metrics)

            self.logger.debug(
                f"📊 交互已记录 - 耗时: {processing_time:.2f}s, "
                f"结果: {'成功' if success else '失败'}"
            )

        except Exception as e:
            self.logger.error(f"❌ 记录交互性能失败: {str(e)}", exc_info=True)

    def get_summary(self) -> Dict[str, Any]:
        """
        获取性能摘要

        Returns:
            性能摘要字典
        """
        # 计算近期性能
        recent_stats = self._calculate_recent_stats()

        return {
            "overall_metrics": self.metrics.to_dict(),
            "recent_performance": recent_stats,
            "monitoring_status": {
                "is_monitoring": self.is_monitoring,
                "monitoring_duration": self._get_monitoring_duration(),
                "total_records": len(self.interaction_history)
            },
            "alerts": {
                "total_alerts": len(self.alerts),
                "recent_alerts": self.alerts[-5:] if self.alerts else []
            }
        }

    def get_all_data(self) -> Dict[str, Any]:
        """
        获取所有性能数据

        Returns:
            完整性能数据字典
        """
        return {
            "metrics": self.metrics.to_dict(),
            "interaction_count": len(self.interaction_history),
            "recent_interactions": [
                record.to_dict()
                for record in list(self.interaction_history)[-20:]
            ],
            "time_series_summary": self._get_time_series_summary(),
            "alerts": self.alerts,
            "resource_usage": list(self.resource_usage_history)
        }

    def start_monitoring(self):
        """启动性能监控"""
        if self.is_monitoring:
            self.logger.warning("⚠️ 性能监控已在运行")
            return

        self.is_monitoring = True
        self.monitoring_start_time = datetime.now()

        self.logger.info("🔍 性能监控已启动")

    def stop_monitoring(self):
        """停止性能监控"""
        if not self.is_monitoring:
            self.logger.warning("⚠️ 性能监控未运行")
            return

        self.is_monitoring = False
        duration = self._get_monitoring_duration()

        self.logger.info(
            f"⏹️ 性能监控已停止 - 监控时长: {duration}"
        )

    def record_resource_usage(self, resource_data: Dict[str, Any]):
        """
        记录资源使用情况

        Args:
            resource_data: 资源使用数据
        """
        self.resource_usage_history.append({
            "timestamp": datetime.now().isoformat(),
            **resource_data
        })

    def get_performance_trend(self) -> Dict[str, Any]:
        """
        获取性能趋势

        Returns:
            性能趋势分析
        """
        if len(self.time_series_data) < 10:
            return {"status": "insufficient_data"}

        # 分析最近的性能趋势
        recent = self.time_series_data[-50:]

        # 计算前后两半的平均处理时间
        mid = len(recent) // 2
        first_half_avg = sum(r['processing_time'] for r in recent[:mid]) / mid
        second_half_avg = sum(r['processing_time'] for r in recent[mid:]) / (len(recent) - mid)

        # 计算成功率趋势
        first_half_success = sum(1 for r in recent[:mid] if r['success']) / mid
        second_half_success = sum(1 for r in recent[mid:] if r['success']) / (len(recent) - mid)

        # 判断趋势
        if second_half_avg < first_half_avg * 0.9:
            speed_trend = "improving"
        elif second_half_avg > first_half_avg * 1.1:
            speed_trend = "declining"
        else:
            speed_trend = "stable"

        if second_half_success > first_half_success + 0.05:
            quality_trend = "improving"
        elif second_half_success < first_half_success - 0.05:
            quality_trend = "declining"
        else:
            quality_trend = "stable"

        return {
            "status": "analyzed",
            "data_points": len(recent),
            "speed_trend": speed_trend,
            "quality_trend": quality_trend,
            "processing_time_change": {
                "first_half_avg": round(first_half_avg, 3),
                "second_half_avg": round(second_half_avg, 3),
                "change_percent": round(
                    ((second_half_avg - first_half_avg) / first_half_avg * 100)
                    if first_half_avg > 0 else 0, 2
                )
            },
            "success_rate_change": {
                "first_half": round(first_half_success, 3),
                "second_half": round(second_half_success, 3),
                "change": round(second_half_success - first_half_success, 3)
            }
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "metrics": self.metrics.to_dict(),
            "history_size": len(self.interaction_history),
            "time_series_size": len(self.time_series_data),
            "alerts_count": len(self.alerts),
            "monitoring_status": self.is_monitoring
        }

    def clear_history(self):
        """清空历史记录"""
        self.interaction_history.clear()
        self.recent_performance.clear()
        self.time_series_data.clear()
        self.alerts.clear()
        self.resource_usage_history.clear()

        # 重置指标
        self.metrics = PerformanceMetrics()

        self.logger.info("🧹 性能历史已清空")

    def _calculate_recent_stats(self) -> Dict[str, Any]:
        """计算近期统计数据"""
        if not self.recent_performance:
            return {"message": "无近期数据"}

        records = list(self.recent_performance)

        # 计算平均处理时间
        avg_time = sum(r.processing_time for r in records) / len(records)

        # 计算成功率
        success_count = sum(1 for r in records if r.success)
        success_rate = success_count / len(records)

        # 计算平均工具调用
        avg_tool_calls = sum(
            r.execution_metrics.get('tool_calls', 0) for r in records
        ) / len(records)

        # 计算平均置信度
        avg_confidence = sum(
            r.cognitive_state.get('confidence_score', 0.0) for r in records
        ) / len(records)

        return {
            "window_size": len(records),
            "average_processing_time": round(avg_time, 3),
            "success_rate": round(success_rate, 3),
            "average_tool_calls": round(avg_tool_calls, 3),
            "average_confidence": round(avg_confidence, 3)
        }

    def _get_time_series_summary(self) -> Dict[str, Any]:
        """获取时间序列摘要"""
        if not self.time_series_data:
            return {"message": "无时间序列数据"}

        return {
            "total_records": len(self.time_series_data),
            "first_record": self.time_series_data[0]["timestamp"],
            "last_record": self.time_series_data[-1]["timestamp"],
            "sample_data": self.time_series_data[-5:]
        }

    def _check_alerts(
        self,
        processing_time: float,
        success: bool,
        execution_metrics: Dict[str, Any]
    ):
        """
        检查是否需要发出告警

        Args:
            processing_time: 处理时间
            success: 是否成功
            execution_metrics: 执行指标
        """
        now = datetime.now()

        # 检查响应时间
        if processing_time > self.alert_threshold_response_time:
            alert = {
                "type": "high_response_time",
                "message": f"响应时间过长: {processing_time:.2f}s (阈值: {self.alert_threshold_response_time}s)",
                "timestamp": now.isoformat(),
                "severity": "warning",
                "value": processing_time
            }
            self.alerts.append(alert)
            self.logger.warning(f"⚠️ 性能告警: {alert['message']}")

        # 检查错误率（基于近期数据）
        if len(self.recent_performance) >= 10:
            recent_errors = sum(
                1 for r in self.recent_performance
                if not r.success
            )
            recent_error_rate = recent_errors / len(self.recent_performance)

            if recent_error_rate > self.alert_threshold_error_rate:
                alert = {
                    "type": "high_error_rate",
                    "message": f"错误率过高: {recent_error_rate:.2%} (阈值: {self.alert_threshold_error_rate:.2%})",
                    "timestamp": now.isoformat(),
                    "severity": "critical",
                    "value": recent_error_rate
                }
                self.alerts.append(alert)
                self.logger.error(f"🚨 严重告警: {alert['message']}")

        # 限制告警数量
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-50:]

    def _get_monitoring_duration(self) -> str:
        """获取监控持续时间"""
        if not self.monitoring_start_time:
            return "N/A"

        if self.is_monitoring:
            duration = datetime.now() - self.monitoring_start_time
        else:
            # 如果已停止，需要从其他地方获取停止时间
            duration = datetime.now() - self.monitoring_start_time

        # 格式化为可读字符串
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours}小时{minutes}分钟{seconds}秒"
        elif minutes > 0:
            return f"{minutes}分钟{seconds}秒"
        else:
            return f"{seconds}秒"
