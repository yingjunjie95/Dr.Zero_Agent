"""
资源优化器 - Resource Optimizer
动态优化系统资源使用，确保Agent在CPU环境下高效运行
提供智能的资源分配、负载平衡和性能调优功能
"""
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque


class OptimizationLevel(Enum):
    """优化级别"""
    CONSERVATIVE = "conservative"  # 保守优化，保持性能
    BALANCED = "balanced"          # 平衡优化
    AGGRESSIVE = "aggressive"      # 激进优化，最大化资源节省


@dataclass
class ResourceAllocation:
    """资源分配方案"""
    timestamp: datetime
    cpu_threads: int
    batch_size: int
    cache_size: int
    memory_limit_mb: int
    optimization_level: OptimizationLevel

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_threads": self.cpu_threads,
            "batch_size": self.batch_size,
            "cache_size": self.cache_size,
            "memory_limit_mb": self.memory_limit_mb,
            "optimization_level": self.optimization_level.value
        }


@dataclass
class OptimizationMetrics:
    """优化指标"""
    total_optimizations: int = 0
    successful_optimizations: int = 0
    failed_optimizations: int = 0
    average_resource_savings: float = 0.0
    performance_impact: float = 0.0

    # 资源节省统计
    cpu_savings_percent: float = 0.0
    memory_savings_percent: float = 0.0

    def update(self, success: bool, resource_savings: float, perf_impact: float):
        """更新优化指标"""
        self.total_optimizations += 1

        if success:
            self.successful_optimizations += 1
        else:
            self.failed_optimizations += 1

        # 更新平均资源节省
        if self.total_optimizations == 1:
            self.average_resource_savings = resource_savings
        else:
            self.average_resource_savings = (
                (self.average_resource_savings * (self.total_optimizations - 1) + resource_savings)
                / self.total_optimizations
            )

        # 更新性能影响
        self.performance_impact = perf_impact

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_optimizations == 0:
            return 0.0
        return self.successful_optimizations / self.total_optimizations

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_optimizations": self.total_optimizations,
            "success_rate": round(self.success_rate, 3),
            "average_resource_savings": round(self.average_resource_savings, 2),
            "performance_impact": round(self.performance_impact, 3),
            "cpu_savings_percent": round(self.cpu_savings_percent, 2),
            "memory_savings_percent": round(self.memory_savings_percent, 2)
        }


class ResourceOptimizer:
    """
    资源优化器

    功能：
    - 动态资源分配：根据负载自动调整资源使用
    - 智能缓存管理：优化缓存大小和策略
    - 负载均衡：平衡多个任务的资源需求
    - 性能调优：在保证性能的前提下最小化资源使用
    - 自适应优化：根据历史数据调整优化策略
    """

    def __init__(
        self,
        max_cpu_threads: int = 4,
        max_memory_mb: int = 8192,
        min_cpu_threads: int = 1,
        min_memory_mb: int = 1024,
        optimization_interval: float = 30.0
    ):
        """
        初始化资源优化器

        Args:
            max_cpu_threads: 最大CPU线程数
            max_memory_mb: 最大内存限制（MB）
            min_cpu_threads: 最小CPU线程数
            min_memory_mb: 最小内存限制（MB）
            optimization_interval: 优化间隔（秒）
        """
        self.max_cpu_threads = max_cpu_threads
        self.max_memory_mb = max_memory_mb
        self.min_cpu_threads = min_cpu_threads
        self.min_memory_mb = min_memory_mb
        self.optimization_interval = optimization_interval

        self.logger = logging.getLogger("ResourceOptimizer")

        # 优化指标
        self.metrics = OptimizationMetrics()

        # 当前资源分配
        self.current_allocation: Optional[ResourceAllocation] = None

        # 资源使用历史
        self.resource_history: deque = deque(maxlen=100)

        # 优化历史
        self.optimization_history: List[Dict[str, Any]] = []

        # 优化策略参数
        self.optimization_level = OptimizationLevel.BALANCED
        self.adaptation_factor = 0.1  # 自适应调整因子

        # 后台优化线程
        self.is_optimizing = False
        self.optimization_thread: Optional[threading.Thread] = None

        # 回调函数
        self.optimization_callbacks: List[callable] = []

        self.logger.info(
            f"✅ 资源优化器初始化完成 - "
            f"CPU: {min_cpu_threads}-{max_cpu_threads}线程, "
            f"内存: {min_memory_mb}-{max_memory_mb}MB"
        )

    def start_optimization(self):
        """启动后台优化"""
        if self.is_optimizing:
            self.logger.warning("⚠️ 资源优化已在运行")
            return

        self.is_optimizing = True
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True,
            name="ResourceOptimizer"
        )
        self.optimization_thread.start()
        self.logger.info("🔄 资源优化已启动")

    def stop_optimization(self):
        """停止优化"""
        self.is_optimizing = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5.0)
        self.logger.info("⏹️ 资源优化已停止")

    def _optimization_loop(self):
        """优化主循环"""
        while self.is_optimizing:
            try:
                self._perform_optimization()
                time.sleep(self.optimization_interval)
            except Exception as e:
                self.logger.error(f"优化循环错误: {str(e)}", exc_info=True)
                time.sleep(self.optimization_interval * 2)

    def _perform_optimization(self):
        """执行一次优化"""
        if not self.resource_history:
            self.logger.debug("无资源历史数据，跳过优化")
            return

        # 分析当前资源使用情况
        analysis = self._analyze_resource_usage()

        # 决定优化策略
        optimization_plan = self._create_optimization_plan(analysis)

        # 执行优化
        if optimization_plan:
            success = self._apply_optimization(optimization_plan)

            # 记录优化结果
            self._record_optimization(optimization_plan, success, analysis)

    def _analyze_resource_usage(self) -> Dict[str, Any]:
        """分析资源使用情况"""
        if not self.resource_history:
            return {}

        recent_data = list(self.resource_history)[-20:]

        # 计算平均值和趋势
        avg_cpu = sum(d.get('cpu_percent', 0) for d in recent_data) / len(recent_data)
        avg_memory = sum(d.get('memory_percent', 0) for d in recent_data) / len(recent_data)

        # 计算趋势（前后两半对比）
        mid = len(recent_data) // 2
        first_half_cpu = sum(d.get('cpu_percent', 0) for d in recent_data[:mid]) / max(mid, 1)
        second_half_cpu = sum(d.get('cpu_percent', 0) for d in recent_data[mid:]) / max(len(recent_data) - mid, 1)

        cpu_trend = "increasing" if second_half_cpu > first_half_cpu * 1.1 else \
                   "decreasing" if second_half_cpu < first_half_cpu * 0.9 else "stable"

        return {
            "avg_cpu_percent": avg_cpu,
            "avg_memory_percent": avg_memory,
            "cpu_trend": cpu_trend,
            "current_load": "high" if avg_cpu > 70 or avg_memory > 80 else
                          "medium" if avg_cpu > 40 or avg_memory > 60 else "low",
            "sample_size": len(recent_data)
        }

    def _create_optimization_plan(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """创建优化计划"""
        if not analysis:
            return None

        current_load = analysis.get('current_load', 'medium')
        cpu_trend = analysis.get('cpu_trend', 'stable')

        # 根据负载和优化级别制定计划
        plan = {
            "timestamp": datetime.now().isoformat(),
            "actions": [],
            "reason": ""
        }

        if current_load == "high":
            # 高负载：减少资源使用
            if self.optimization_level in [OptimizationLevel.BALANCED, OptimizationLevel.AGGRESSIVE]:
                new_threads = max(self.min_cpu_threads, self._get_current_threads() - 1)
                new_batch_size = max(1, self._get_current_batch_size() - 1)

                plan["actions"].append({
                    "type": "reduce_threads",
                    "current": self._get_current_threads(),
                    "target": new_threads
                })
                plan["actions"].append({
                    "type": "reduce_batch_size",
                    "current": self._get_current_batch_size(),
                    "target": new_batch_size
                })
                plan["reason"] = "高负载，减少资源使用"

        elif current_load == "low" and cpu_trend == "stable":
            # 低负载且稳定：可以适当增加资源提升性能
            if self.optimization_level == OptimizationLevel.CONSERVATIVE:
                new_threads = min(self.max_cpu_threads, self._get_current_threads() + 1)
                plan["actions"].append({
                    "type": "increase_threads",
                    "current": self._get_current_threads(),
                    "target": new_threads
                })
                plan["reason"] = "低负载，增加资源提升性能"

        elif cpu_trend == "increasing":
            # CPU使用率上升：预防性优化
            plan["actions"].append({
                "type": "enable_cache_optimization",
                "description": "启用缓存优化以应对上升的负载"
            })
            plan["reason"] = "CPU使用率上升趋势，预防性优化"

        if not plan["actions"]:
            return None

        return plan

    def _apply_optimization(self, plan: Dict[str, Any]) -> bool:
        """应用优化计划"""
        try:
            self.logger.info(f"🔧 应用优化计划: {plan['reason']}")

            for action in plan['actions']:
                action_type = action['type']

                if action_type == "reduce_threads":
                    self._adjust_thread_count(action['target'])
                elif action_type == "increase_threads":
                    self._adjust_thread_count(action['target'])
                elif action_type == "reduce_batch_size":
                    self._adjust_batch_size(action['target'])
                elif action_type == "enable_cache_optimization":
                    self._optimize_cache()

                self.logger.debug(f"  ✅ 执行动作: {action_type}")

            # 触发回调
            for callback in self.optimization_callbacks:
                try:
                    callback(plan)
                except Exception as e:
                    self.logger.error(f"优化回调执行失败: {str(e)}")

            return True

        except Exception as e:
            self.logger.error(f"应用优化计划失败: {str(e)}", exc_info=True)
            return False

    def record_resource_usage(self, usage_data: Dict[str, Any]):
        """
        记录资源使用数据

        Args:
            usage_data: 资源使用数据，包含cpu_percent, memory_percent等
        """
        self.resource_history.append({
            "timestamp": datetime.now().isoformat(),
            **usage_data
        })

    def set_optimization_level(self, level: OptimizationLevel):
        """设置优化级别"""
        old_level = self.optimization_level
        self.optimization_level = level
        self.logger.info(f"🎯 优化级别已更改: {old_level.value} → {level.value}")

    def add_optimization_callback(self, callback: callable):
        """添加优化回调函数"""
        self.optimization_callbacks.append(callback)

    def get_current_allocation(self) -> Optional[Dict[str, Any]]:
        """获取当前资源分配"""
        if self.current_allocation:
            return self.current_allocation.to_dict()
        return None

    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        return {
            "metrics": self.metrics.to_dict(),
            "current_allocation": self.get_current_allocation(),
            "optimization_level": self.optimization_level.value,
            "total_optimizations": len(self.optimization_history),
            "recent_optimizations": self.optimization_history[-5:]
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "metrics": self.metrics.to_dict(),
            "resource_history_size": len(self.resource_history),
            "optimization_history_size": len(self.optimization_history),
            "is_optimizing": self.is_optimizing,
            "optimization_level": self.optimization_level.value
        }

    def _adjust_thread_count(self, new_count: int):
        """调整线程数（实际项目中需要与线程池集成）"""
        self.logger.debug(f"调整线程数: {new_count}")
        # TODO: 与实际线程池集成

    def _adjust_batch_size(self, new_size: int):
        """调整批处理大小"""
        self.logger.debug(f"调整批处理大小: {new_size}")
        # TODO: 与模型推理模块集成

    def _optimize_cache(self):
        """优化缓存"""
        self.logger.debug("执行缓存优化")
        # TODO: 与缓存系统集成

    def _get_current_threads(self) -> int:
        """获取当前线程数"""
        # TODO: 从实际配置中获取
        return self.max_cpu_threads

    def _get_current_batch_size(self) -> int:
        """获取当前批处理大小"""
        # TODO: 从实际配置中获取
        return 2

    def _record_optimization(
        self,
        plan: Dict[str, Any],
        success: bool,
        analysis: Dict[str, Any]
    ):
        """记录优化结果"""
        # 更新指标
        resource_savings = 5.0 if success else 0.0  # 示例值
        perf_impact = -0.02 if success else 0.0  # 示例值
        self.metrics.update(success, resource_savings, perf_impact)

        # 记录历史
        record = {
            "timestamp": datetime.now().isoformat(),
            "plan": plan,
            "success": success,
            "analysis": analysis,
            "resource_savings": resource_savings,
            "performance_impact": perf_impact
        }
        self.optimization_history.append(record)

        # 限制历史记录大小
        if len(self.optimization_history) > 50:
            self.optimization_history.pop(0)

        status = "✅" if success else "❌"
        self.logger.info(f"{status} 优化已完成 - {plan['reason']}")

    def reset_statistics(self):
        """重置统计数据"""
        self.metrics = OptimizationMetrics()
        self.optimization_history.clear()
        self.resource_history.clear()
        self.logger.info("🧹 优化统计数据已重置")
