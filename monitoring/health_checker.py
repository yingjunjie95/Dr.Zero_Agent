"""
健康检查器 - Health Checker
全面监控系统各组件的健康状态，提供故障检测和自动恢复建议
确保Agent稳定运行，及时发现并处理潜在问题
"""
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque


class HealthStatus(Enum):
    """健康状态等级"""
    HEALTHY = "healthy"           # 健康
    DEGRADED = "degraded"         # 性能下降
    WARNING = "warning"           # 警告
    CRITICAL = "critical"         # 严重
    UNAVAILABLE = "unavailable"   # 不可用


@dataclass
class ComponentHealth:
    """组件健康状态"""
    name: str
    status: HealthStatus
    last_check: datetime
    response_time_ms: float = 0.0
    error_count: int = 0
    details: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "status": self.status.value,
            "last_check": self.last_check.isoformat(),
            "response_time_ms": round(self.response_time_ms, 2),
            "error_count": self.error_count,
            "details": self.details,
            "metrics": self.metrics
        }


@dataclass
class HealthReport:
    """健康检查报告"""
    timestamp: datetime
    overall_status: HealthStatus
    component_healths: List[ComponentHealth]
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    check_duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_status": self.overall_status.value,
            "components": [ch.to_dict() for ch in self.component_healths],
            "issues_count": len(self.issues),
            "issues": self.issues[:10],  # 最多返回10个问题
            "recommendations": self.recommendations,
            "check_duration_ms": round(self.check_duration_ms, 2)
        }


class HealthChecker:
    """
    健康检查器

    功能：
    - 组件健康监控：定期检查各组件运行状态
    - 故障检测：识别系统异常和性能瓶颈
    - 自动诊断：分析问题原因并提供解决方案
    - 趋势分析：跟踪健康状态变化趋势
    - 告警通知：在检测到严重问题时发出告警
    - 恢复建议：提供自动或手动恢复建议
    """

    def __init__(
        self,
        check_interval: float = 60.0,
        response_timeout_ms: float = 5000.0,
        max_error_threshold: int = 5
    ):
        """
        初始化健康检查器

        Args:
            check_interval: 检查间隔（秒）
            response_timeout_ms: 响应超时阈值（毫秒）
            max_error_threshold: 最大错误次数阈值
        """
        self.check_interval = check_interval
        self.response_timeout_ms = response_timeout_ms
        self.max_error_threshold = max_error_threshold

        self.logger = logging.getLogger("HealthChecker")

        # 组件注册表
        self.registered_components: Dict[str, Any] = {}

        # 健康历史
        self.health_history: deque = deque(maxlen=100)

        # 问题历史
        self.issues_history: List[Dict[str, Any]] = []

        # 检查统计
        self.total_checks = 0
        self.failed_checks = 0

        # 后台检查线程
        self.is_checking = False
        self.check_thread: Optional[threading.Thread] = None

        # 告警回调
        self.alert_callbacks: List[callable] = []

        # 最后一次健康报告
        self.last_report: Optional[HealthReport] = None

        self.logger.info(
            f"✅ 健康检查器初始化完成 - 检查间隔: {check_interval}s"
        )

    def register_component(
        self,
        name: str,
        health_check_func: callable,
        critical: bool = True
    ):
        """
        注册需要监控的组件

        Args:
            name: 组件名称
            health_check_func: 健康检查函数，返回Dict[str, Any]
            critical: 是否为关键组件
        """
        self.registered_components[name] = {
            "health_check": health_check_func,
            "critical": critical,
            "consecutive_failures": 0,
            "last_success": None
        }
        self.logger.info(f"📦 已注册组件: {name} (关键: {critical})")

    def start_monitoring(self):
        """启动后台健康检查"""
        if self.is_checking:
            self.logger.warning("⚠️ 健康检查已在运行")
            return

        self.is_checking = True
        self.check_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="HealthChecker"
        )
        self.check_thread.start()
        self.logger.info("🔍 健康检查已启动")

    def stop_monitoring(self):
        """停止健康检查"""
        self.is_checking = False
        if self.check_thread:
            self.check_thread.join(timeout=5.0)
        self.logger.info("⏹️ 健康检查已停止")

    def _monitoring_loop(self):
        """监控主循环"""
        while self.is_checking:
            try:
                self.perform_health_check()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"健康检查循环错误: {str(e)}", exc_info=True)
                time.sleep(self.check_interval * 2)

    def perform_health_check(self) -> HealthReport:
        """
        执行健康检查

        Returns:
            健康检查报告
        """
        start_time = time.time()
        self.total_checks += 1

        try:
            # 检查所有组件
            component_healths = []
            issues = []
            recommendations = []

            for name, component in self.registered_components.items():
                health = self._check_component(name, component)
                component_healths.append(health)

                # 收集问题和推荐
                if health.status != HealthStatus.HEALTHY:
                    issue = {
                        "component": name,
                        "status": health.status.value,
                        "details": health.details,
                        "timestamp": datetime.now().isoformat()
                    }
                    issues.append(issue)
                    self.issues_history.append(issue)

                    # 生成推荐
                    recs = self._generate_recommendations(name, health)
                    recommendations.extend(recs)

            # 确定整体状态
            overall_status = self._determine_overall_status(component_healths)

            # 计算检查耗时
            check_duration_ms = (time.time() - start_time) * 1000

            # 创建报告
            report = HealthReport(
                timestamp=datetime.now(),
                overall_status=overall_status,
                component_healths=component_healths,
                issues=issues,
                recommendations=list(set(recommendations)),  # 去重
                check_duration_ms=check_duration_ms
            )

            # 保存报告
            self.last_report = report
            self.health_history.append(report)

            # 限制问题历史大小
            if len(self.issues_history) > 200:
                self.issues_history = self.issues_history[-100:]

            # 记录日志
            self._log_health_report(report)

            # 触发告警（如果有严重问题）
            if overall_status in [HealthStatus.CRITICAL, HealthStatus.UNAVAILABLE]:
                self._trigger_alert(report)

            return report

        except Exception as e:
            self.failed_checks += 1
            self.logger.error(f"❌ 健康检查失败: {str(e)}", exc_info=True)

            # 返回失败报告
            check_duration_ms = (time.time() - start_time) * 1000
            return HealthReport(
                timestamp=datetime.now(),
                overall_status=HealthStatus.CRITICAL,
                component_healths=[],
                issues=[{
                    "component": "HealthChecker",
                    "status": "error",
                    "details": f"健康检查执行失败: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }],
                recommendations=["检查健康检查器配置和依赖"],
                check_duration_ms=check_duration_ms
            )

    def _check_component(self, name: str, component: Dict[str, Any]) -> ComponentHealth:
        """检查单个组件的健康状态"""
        try:
            # 执行健康检查函数
            check_result = component["health_check"]()

            # 解析检查结果
            status = self._parse_health_status(check_result)
            response_time = check_result.get('response_time_ms', 0.0)
            error_count = check_result.get('error_count', 0)
            details = check_result.get('details', '')
            metrics = check_result.get('metrics', {})

            # 更新组件状态
            if status == HealthStatus.HEALTHY:
                component["consecutive_failures"] = 0
                component["last_success"] = datetime.now()
            else:
                component["consecutive_failures"] += 1

                # 检查是否超过阈值
                if component["consecutive_failures"] >= self.max_error_threshold:
                    status = HealthStatus.CRITICAL
                    details += f" | 连续失败{component['consecutive_failures']}次"

            return ComponentHealth(
                name=name,
                status=status,
                last_check=datetime.now(),
                response_time_ms=response_time,
                error_count=error_count,
                details=details,
                metrics=metrics
            )

        except Exception as e:
            component["consecutive_failures"] += 1
            self.logger.error(f"组件 {name} 检查失败: {str(e)}")

            return ComponentHealth(
                name=name,
                status=HealthStatus.UNAVAILABLE,
                last_check=datetime.now(),
                response_time_ms=0.0,
                error_count=component["consecutive_failures"],
                details=f"检查异常: {str(e)}"
            )

    def _parse_health_status(self, check_result: Dict[str, Any]) -> HealthStatus:
        """解析健康状态"""
        status_str = check_result.get('status', 'unknown').lower()

        status_map = {
            'healthy': HealthStatus.HEALTHY,
            'ok': HealthStatus.HEALTHY,
            'good': HealthStatus.HEALTHY,
            'degraded': HealthStatus.DEGRADED,
            'warning': HealthStatus.WARNING,
            'warn': HealthStatus.WARNING,
            'critical': HealthStatus.CRITICAL,
            'error': HealthStatus.CRITICAL,
            'unavailable': HealthStatus.UNAVAILABLE,
            'down': HealthStatus.UNAVAILABLE
        }

        return status_map.get(status_str, HealthStatus.WARNING)

    def _determine_overall_status(self, component_healths: List[ComponentHealth]) -> HealthStatus:
        """确定整体健康状态"""
        if not component_healths:
            return HealthStatus.UNAVAILABLE

        # 检查是否有不可用的关键组件
        for health in component_healths:
            component = self.registered_components.get(health.name, {})
            if component.get('critical', False) and health.status == HealthStatus.UNAVAILABLE:
                return HealthStatus.UNAVAILABLE

        # 检查是否有严重状态的组件
        critical_count = sum(1 for h in component_healths if h.status == HealthStatus.CRITICAL)
        if critical_count > 0:
            return HealthStatus.CRITICAL

        # 检查是否有警告状态的组件
        warning_count = sum(1 for h in component_healths if h.status in [HealthStatus.WARNING, HealthStatus.DEGRADED])
        if warning_count > 0:
            return HealthStatus.WARNING

        return HealthStatus.HEALTHY

    def _generate_recommendations(self, component_name: str, health: ComponentHealth) -> List[str]:
        """生成修复建议"""
        recommendations = []

        if health.status == HealthStatus.CRITICAL:
            recommendations.append(f"🚨 紧急: {component_name} 处于严重状态，需要立即检查")

            if health.response_time_ms > self.response_timeout_ms:
                recommendations.append(f"⏱️ {component_name} 响应时间过长({health.response_time_ms:.0f}ms)，考虑优化或重启")

            if health.error_count > self.max_error_threshold:
                recommendations.append(f"❌ {component_name} 错误次数过多({health.error_count})，检查日志定位问题")

        elif health.status == HealthStatus.WARNING:
            recommendations.append(f"⚠️ {component_name} 存在警告，建议监控其状态变化")

            if health.response_time_ms > self.response_timeout_ms * 0.7:
                recommendations.append(f"⏱️ {component_name} 响应时间接近阈值，可能需要优化")

        elif health.status == HealthStatus.DEGRADED:
            recommendations.append(f"📉 {component_name} 性能下降，检查资源使用情况")

        return recommendations

    def _log_health_report(self, report: HealthReport):
        """记录健康报告日志"""
        status_emoji = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.DEGRADED: "📉",
            HealthStatus.WARNING: "⚠️",
            HealthStatus.CRITICAL: "🚨",
            HealthStatus.UNAVAILABLE: "❌"
        }

        emoji = status_emoji.get(report.overall_status, "❓")
        self.logger.info(
            f"{emoji} 健康检查完成 - 状态: {report.overall_status.value}, "
            f"组件数: {len(report.component_healths)}, "
            f"问题数: {len(report.issues)}, "
            f"耗时: {report.check_duration_ms:.0f}ms"
        )

        # 记录详细问题
        if report.issues:
            for issue in report.issues[:3]:  # 只记录前3个问题
                self.logger.warning(
                    f"  问题: {issue['component']} - {issue['details']}"
                )

    def _trigger_alert(self, report: HealthReport):
        """触发告警"""
        alert_data = {
            "type": "health_critical",
            "timestamp": datetime.now().isoformat(),
            "overall_status": report.overall_status.value,
            "issues_count": len(report.issues),
            "report": report.to_dict()
        }

        self.logger.critical(
            f"🚨 严重健康告警 - 状态: {report.overall_status.value}, "
            f"问题数: {len(report.issues)}"
        )

        # 调用告警回调
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                self.logger.error(f"告警回调执行失败: {str(e)}")

    def add_alert_callback(self, callback: callable):
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)

    def get_current_health(self) -> Optional[Dict[str, Any]]:
        """获取当前健康状态"""
        if self.last_report:
            return self.last_report.to_dict()
        return None

    def get_health_trend(self, window: int = 10) -> Dict[str, Any]:
        """
        获取健康趋势

        Args:
            window: 时间窗口大小

        Returns:
            健康趋势分析
        """
        if len(self.health_history) < window:
            return {"status": "insufficient_data"}

        recent_reports = list(self.health_history)[-window:]

        # 统计各状态出现次数
        status_counts = {
            "healthy": 0,
            "degraded": 0,
            "warning": 0,
            "critical": 0,
            "unavailable": 0
        }

        for report in recent_reports:
            status_counts[report.overall_status.value] += 1

        # 判断趋势
        if status_counts["critical"] > 0 or status_counts["unavailable"] > 0:
            trend = "declining"
        elif status_counts["healthy"] == window:
            trend = "stable"
        elif status_counts["healthy"] > window * 0.7:
            trend = "stable"
        else:
            trend = "fluctuating"

        return {
            "status": "analyzed",
            "trend": trend,
            "window_size": window,
            "status_distribution": status_counts,
            "health_score": round(
                (status_counts["healthy"] / window) * 100, 2
            )
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_checks": self.total_checks,
            "failed_checks": self.failed_checks,
            "success_rate": round(
                (self.total_checks - self.failed_checks) / max(self.total_checks, 1) * 100, 2
            ),
            "registered_components": len(self.registered_components),
            "issues_history_size": len(self.issues_history),
            "is_checking": self.is_checking,
            "last_check": self.last_report.timestamp.isoformat() if self.last_report else None
        }

    def reset_statistics(self):
        """重置统计数据"""
        self.total_checks = 0
        self.failed_checks = 0
        self.issues_history.clear()
        self.health_history.clear()
        self.last_report = None

        # 重置组件失败计数
        for component in self.registered_components.values():
            component["consecutive_failures"] = 0

        self.logger.info("🧹 健康检查统计数据已重置")
