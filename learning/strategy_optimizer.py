"""
策略优化器 - Strategy Optimizer
基于历史经验和性能数据，持续优化Agent的决策策略
实现自适应的策略调整、参数调优和模式学习
专为CPU环境优化，提供高效的策略改进建议
"""
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import os


class OptimizationType(Enum):
    """优化类型"""
    STRATEGY_ADJUSTMENT = "strategy_adjustment"  # 策略调整
    PARAMETER_TUNING = "parameter_tuning"  # 参数调优
    TOOL_SELECTION = "tool_selection"  # 工具选择优化
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"  # 阈值调整
    PATTERN_LEARNING = "pattern_learning"  # 模式学习


class OptimizationPriority(Enum):
    """优化优先级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class OptimizationSuggestion:
    """优化建议"""
    suggestion_id: str
    optimization_type: OptimizationType
    priority: OptimizationPriority
    description: str
    current_value: Any
    suggested_value: Any
    expected_improvement: float  # 预期改进幅度 (0.0-1.0)
    confidence: float  # 建议置信度 (0.0-1.0)

    # 支持数据
    supporting_evidence: List[str] = field(default_factory=list)
    implementation_steps: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)

    # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    applied: bool = False
    application_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "suggestion_id": self.suggestion_id,
            "optimization_type": self.optimization_type.value,
            "priority": self.priority.value,
            "description": self.description,
            "current_value": str(self.current_value)[:200],
            "suggested_value": str(self.suggested_value)[:200],
            "expected_improvement": round(self.expected_improvement, 3),
            "confidence": round(self.confidence, 3),
            "supporting_evidence": self.supporting_evidence[:5],
            "implementation_steps": self.implementation_steps,
            "risks": self.risks,
            "created_at": self.created_at.isoformat(),
            "applied": self.applied,
            "application_result": self.application_result
        }


@dataclass
class StrategyProfile:
    """策略档案"""
    strategy_name: str
    category: str  # 策略类别

    # 性能指标
    total_uses: int = 0
    success_count: int = 0
    failure_count: int = 0
    average_reward: float = 0.0
    average_execution_time: float = 0.0

    # 适用场景
    applicable_scenarios: List[str] = field(default_factory=list)
    best_conditions: Dict[str, Any] = field(default_factory=dict)
    worst_conditions: Dict[str, Any] = field(default_factory=dict)

    # 时间信息
    first_used: Optional[datetime] = None
    last_used: Optional[datetime] = None
    last_optimized: Optional[datetime] = None

    # 趋势
    recent_performance: deque = field(default_factory=lambda: deque(maxlen=20))
    performance_trend: str = "stable"  # improving, declining, stable, volatile

    def update_performance(self, success: bool, reward: float, execution_time: float):
        """更新性能记录"""
        self.total_uses += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        # 更新平均奖励（移动平均）
        self.average_reward = (
            (self.average_reward * (self.total_uses - 1) + reward) / self.total_uses
        )

        # 更新平均执行时间
        self.average_execution_time = (
            (self.average_execution_time * (self.total_uses - 1) + execution_time)
            / self.total_uses
        )

        # 更新时间
        now = datetime.now()
        if not self.first_used:
            self.first_used = now
        self.last_used = now

        # 记录最近表现
        self.recent_performance.append({
            "timestamp": now,
            "success": success,
            "reward": reward,
            "execution_time": execution_time
        })

        # 更新趋势
        self._update_trend()

    def _update_trend(self):
        """更新性能趋势"""
        if len(self.recent_performance) < 5:
            self.performance_trend = "insufficient_data"
            return

        recent = list(self.recent_performance)

        # 计算前后两半的成功率
        mid = len(recent) // 2
        first_half_success = sum(1 for r in recent[:mid] if r["success"]) / mid
        second_half_success = sum(1 for r in recent[mid:] if r["success"]) / (len(recent) - mid)

        change = second_half_success - first_half_success

        if change > 0.15:
            self.performance_trend = "improving"
        elif change < -0.15:
            self.performance_trend = "declining"
        elif abs(change) < 0.05:
            self.performance_trend = "stable"
        else:
            self.performance_trend = "volatile"

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_uses == 0:
            return 0.0
        return self.success_count / self.total_uses

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "strategy_name": self.strategy_name,
            "category": self.category,
            "total_uses": self.total_uses,
            "success_rate": round(self.success_rate, 3),
            "average_reward": round(self.average_reward, 3),
            "average_execution_time": round(self.average_execution_time, 3),
            "performance_trend": self.performance_trend,
            "applicable_scenarios": self.applicable_scenarios,
            "last_optimized": self.last_optimized.isoformat() if self.last_optimized else None
        }


@dataclass
class OptimizerConfig:
    """优化器配置"""
    min_data_points: int = 20  # 开始优化的最小数据点
    optimization_interval_seconds: int = 600  # 优化间隔（秒）
    max_suggestions_per_cycle: int = 10  # 每次优化最大建议数

    # 学习率
    learning_rate: float = 0.1  # 策略调整学习率
    exploration_rate: float = 0.2  # 探索率（尝试新策略的概率）

    # 阈值
    improvement_threshold: float = 0.05  # 显著改进阈值
    decline_threshold: float = -0.1  # 显著下降阈值

    # 持久化
    persistence_enabled: bool = True
    storage_path: str = "data/strategy_optimizer"
    save_interval: int = 50  # 保存间隔


class StrategyOptimizer:
    """
    策略优化器

    功能：
    - 策略性能追踪：监控各策略的表现
    - 模式识别：发现成功和失败的模式
    - 参数调优：自动调整决策阈值和参数
    - 工具优化：优化工具选择和使用策略
    - 建议生成：提供可操作的优化建议
    - 自适应学习：根据反馈持续改进
    """

    def __init__(self, config: Optional[OptimizerConfig] = None):
        """
        初始化策略优化器

        Args:
            config: 优化器配置
        """
        self.config = config or OptimizerConfig()
        self.logger = logging.getLogger("StrategyOptimizer")

        # 策略档案库
        self.strategy_profiles: Dict[str, StrategyProfile] = {}

        # 优化历史
        self.optimization_history: List[Dict[str, Any]] = []
        self.applied_suggestions: List[OptimizationSuggestion] = []

        # 性能基线
        self.performance_baseline = {
            "average_success_rate": 0.0,
            "average_reward": 0.0,
            "average_response_time": 0.0
        }

        # 统计信息
        self.stats = {
            "total_interactions_analyzed": 0,
            "total_optimizations_performed": 0,
            "total_suggestions_generated": 0,
            "suggestions_applied": 0,
            "successful_optimizations": 0,
            "failed_optimizations": 0
        }

        # 上次优化时间
        self.last_optimization_time: Optional[datetime] = None

        # 创建存储目录
        if self.config.persistence_enabled:
            os.makedirs(self.config.storage_path, exist_ok=True)

        # 加载已有数据
        if self.config.persistence_enabled:
            self._load_from_disk()

        self.logger.info(f"✅ 策略优化器初始化完成")

    def record_interaction(
        self,
        strategy_name: str,
        strategy_category: str,
        success: bool,
        reward: float,
        execution_time: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        记录交互结果

        Args:
            strategy_name: 使用的策略名称
            strategy_category: 策略类别
            success: 是否成功
            reward: 奖励信号
            execution_time: 执行时间
            context: 上下文信息
        """
        # 获取或创建策略档案
        if strategy_name not in self.strategy_profiles:
            self.strategy_profiles[strategy_name] = StrategyProfile(
                strategy_name=strategy_name,
                category=strategy_category
            )

        profile = self.strategy_profiles[strategy_name]

        # 更新性能
        profile.update_performance(success, reward, execution_time)

        # 更新场景信息
        if context:
            self._update_scenario_info(profile, context, success)

        # 更新统计
        self.stats["total_interactions_analyzed"] += 1

        # 更新基线
        self._update_performance_baseline()

        self.logger.debug(
            f"📊 记录交互: {strategy_name} - "
            f"成功: {success}, 奖励: {reward:.2f}"
        )

    def analyze_and_optimize(self) -> Dict[str, Any]:
        """
        执行分析和优化

        Returns:
            优化结果
        """
        now = datetime.now()

        # 检查是否需要优化
        if not self._should_optimize(now):
            return {
                "status": "not_ready",
                "reason": "未达到优化间隔或数据不足",
                "next_optimization_in": self._time_to_next_optimization(now)
            }

        self.logger.info("🔍 开始策略分析和优化...")

        optimization_result = {
            "timestamp": now.isoformat(),
            "strategies_analyzed": len(self.strategy_profiles),
            "suggestions_generated": [],
            "performance_summary": self._get_performance_summary(),
            "optimization_focus_areas": []
        }

        try:
            # 1. 识别需要优化的策略
            focus_strategies = self._identify_focus_strategies()
            optimization_result["optimization_focus_areas"] = [
                s.strategy_name for s in focus_strategies
            ]

            # 2. 生成优化建议
            suggestions = []

            # 2.1 策略调整建议
            strategy_suggestions = self._generate_strategy_adjustments(focus_strategies)
            suggestions.extend(strategy_suggestions)

            # 2.2 参数调优建议
            parameter_suggestions = self._generate_parameter_tuning_suggestions()
            suggestions.extend(parameter_suggestions)

            # 2.3 工具选择优化
            tool_suggestions = self._generate_tool_optimization_suggestions()
            suggestions.extend(tool_suggestions)

            # 2.4 阈值调整建议
            threshold_suggestions = self._generate_threshold_adjustments()
            suggestions.extend(threshold_suggestions)

            # 3. 排序和限制建议数量
            suggestions.sort(key=lambda s: (
                {"critical": 4, "high": 3, "medium": 2, "low": 1}[s.priority.value] * 1000 +
                s.expected_improvement * 100
            ), reverse=True)

            top_suggestions = suggestions[:self.config.max_suggestions_per_cycle]

            # 4. 记录建议
            for suggestion in top_suggestions:
                optimization_result["suggestions_generated"].append(suggestion.to_dict())
                self.applied_suggestions.append(suggestion)
                self.stats["total_suggestions_generated"] += 1

            # 5. 更新优化历史
            self.optimization_history.append({
                "timestamp": now.isoformat(),
                "strategies_analyzed": len(self.strategy_profiles),
                "suggestions_count": len(top_suggestions),
                "focus_areas": optimization_result["optimization_focus_areas"]
            })

            self.stats["total_optimizations_performed"] += 1
            self.last_optimization_time = now

            # 6. 定期保存
            if self.config.persistence_enabled:
                if self.stats["total_optimizations_performed"] % 5 == 0:
                    self._save_to_disk()

            self.logger.info(
                f"✨ 优化完成 - 生成 {len(top_suggestions)} 条建议"
            )

            return optimization_result

        except Exception as e:
            self.logger.error(f"❌ 优化过程出错: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }

    def apply_suggestion(self, suggestion_id: str, result: Dict[str, Any]) -> bool:
        """
        应用优化建议并记录结果

        Args:
            suggestion_id: 建议ID
            result: 应用结果

        Returns:
            是否成功应用
        """
        for suggestion in self.applied_suggestions:
            if suggestion.suggestion_id == suggestion_id:
                suggestion.applied = True
                suggestion.application_result = result

                if result.get("success", False):
                    self.stats["suggestions_applied"] += 1
                    self.stats["successful_optimizations"] += 1
                else:
                    self.stats["failed_optimizations"] += 1

                self.logger.info(
                    f"✅ 建议已应用: {suggestion_id} - "
                    f"结果: {result.get('success', False)}"
                )

                return True

        self.logger.warning(f"⚠️ 未找到建议: {suggestion_id}")
        return False

    def get_strategy_performance(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        获取策略性能

        Args:
            strategy_name: 策略名称

        Returns:
            策略性能数据
        """
        if strategy_name in self.strategy_profiles:
            return self.strategy_profiles[strategy_name].to_dict()
        return None

    def get_all_strategies_performance(self) -> Dict[str, Dict[str, Any]]:
        """获取所有策略的性能"""
        return {
            name: profile.to_dict()
            for name, profile in self.strategy_profiles.items()
        }

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """获取优化推荐"""
        if not self.applied_suggestions:
            return []

        # 返回最近未应用的建议
        unapplied = [
            s for s in self.applied_suggestions
            if not s.applied
        ]

        unapplied.sort(key=lambda s: s.created_at, reverse=True)

        return [s.to_dict() for s in unapplied[:10]]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "strategy_count": len(self.strategy_profiles),
            "performance_baseline": self.performance_baseline,
            "last_optimization": (
                self.last_optimization_time.isoformat()
                if self.last_optimization_time else None
            ),
            "optimization_history_size": len(self.optimization_history)
        }

    def _should_optimize(self, now: datetime) -> bool:
        """判断是否应该进行优化"""
        # 检查数据量
        if self.stats["total_interactions_analyzed"] < self.config.min_data_points:
            return False

        # 检查时间间隔
        if self.last_optimization_time is None:
            return True

        elapsed = (now - self.last_optimization_time).total_seconds()
        return elapsed >= self.config.optimization_interval_seconds

    def _time_to_next_optimization(self, now: datetime) -> float:
        """计算距离下次优化的时间（秒）"""
        if self.last_optimization_time is None:
            return 0.0

        elapsed = (now - self.last_optimization_time).total_seconds()
        remaining = self.config.optimization_interval_seconds - elapsed

        return max(0.0, remaining)

    def _identify_focus_strategies(self) -> List[StrategyProfile]:
        """识别需要重点关注的策略"""
        focus_strategies = []

        for profile in self.strategy_profiles.values():
            # 跳过使用次数太少的策略
            if profile.total_uses < 5:
                continue

            # 关注以下情况：
            # 1. 性能下降
            if profile.performance_trend == "declining":
                focus_strategies.append(profile)

            # 2. 高波动性
            elif profile.performance_trend == "volatile":
                focus_strategies.append(profile)

            # 3. 成功率过低
            elif profile.success_rate < 0.5:
                focus_strategies.append(profile)

            # 4. 高使用频率但表现一般
            elif profile.total_uses > 50 and profile.success_rate < 0.7:
                focus_strategies.append(profile)

        return focus_strategies

    def _generate_strategy_adjustments(
        self,
        focus_strategies: List[StrategyProfile]
    ) -> List[OptimizationSuggestion]:
        """生成策略调整建议"""
        import uuid
        suggestions = []

        for profile in focus_strategies:
            # 针对性能下降的策略
            if profile.performance_trend == "declining":
                suggestion = OptimizationSuggestion(
                    suggestion_id=f"opt_{uuid.uuid4().hex[:8]}",
                    optimization_type=OptimizationType.STRATEGY_ADJUSTMENT,
                    priority=OptimizationPriority.HIGH,
                    description=f"策略 '{profile.strategy_name}' 性能下降，建议调整",
                    current_value=f"成功率: {profile.success_rate:.2%}",
                    suggested_value="采用更保守的执行方式或增加验证步骤",
                    expected_improvement=0.15,
                    confidence=0.7,
                    supporting_evidence=[
                        f"最近{len(profile.recent_performance)}次使用中性能下降",
                        f"当前成功率: {profile.success_rate:.2%}",
                        f"平均奖励: {profile.average_reward:.2f}"
                    ],
                    implementation_steps=[
                        "1. 分析最近失败的案例",
                        "2. 识别共同的失败模式",
                        "3. 增加前置条件检查",
                        "4. 添加回退机制"
                    ],
                    risks=["可能降低响应速度", "可能过于保守"]
                )
                suggestions.append(suggestion)

            # 针对高波动性的策略
            elif profile.performance_trend == "volatile":
                suggestion = OptimizationSuggestion(
                    suggestion_id=f"opt_{uuid.uuid4().hex[:8]}",
                    optimization_type=OptimizationType.PARAMETER_TUNING,
                    priority=OptimizationPriority.MEDIUM,
                    description=f"策略 '{profile.strategy_name}' 表现不稳定，建议稳定化",
                    current_value=f"波动性高",
                    suggested_value="标准化输入处理和输出格式",
                    expected_improvement=0.1,
                    confidence=0.6,
                    supporting_evidence=[
                        "性能波动较大，不一致",
                        f"使用次数: {profile.total_uses}"
                    ],
                    implementation_steps=[
                        "1. 统一输入预处理流程",
                        "2. 标准化错误处理",
                        "3. 添加一致性检查"
                    ],
                    risks=["可能降低灵活性"]
                )
                suggestions.append(suggestion)

        return suggestions

    def _generate_parameter_tuning_suggestions(self) -> List[OptimizationSuggestion]:
        """生成参数调优建议"""
        import uuid
        suggestions = []

        # 基于整体性能基线
        if self.performance_baseline["average_success_rate"] < 0.6:
            suggestion = OptimizationSuggestion(
                suggestion_id=f"opt_{uuid.uuid4().hex[:8]}",
                optimization_type=OptimizationType.PARAMETER_TUNING,
                priority=OptimizationPriority.HIGH,
                description="整体成功率偏低，建议调整置信度阈值",
                current_value=f"平均成功率: {self.performance_baseline['average_success_rate']:.2%}",
                suggested_value="提高决策置信度阈值至0.6-0.7",
                expected_improvement=0.12,
                confidence=0.65,
                supporting_evidence=[
                    f"当前平均成功率: {self.performance_baseline['average_success_rate']:.2%}",
                    "低于目标阈值60%"
                ],
                implementation_steps=[
                    "1. 调整决策器的confidence_threshold",
                    "2. 增加澄清询问的频率",
                    "3. 强化风险评估"
                ],
                risks=["可能导致更多澄清询问", "用户体验可能受影响"]
            )
            suggestions.append(suggestion)

        # 基于响应时间
        if self.performance_baseline["average_response_time"] > 3.0:
            suggestion = OptimizationSuggestion(
                suggestion_id=f"opt_{uuid.uuid4().hex[:8]}",
                optimization_type=OptimizationType.PARAMETER_TUNING,
                priority=OptimizationPriority.MEDIUM,
                description="平均响应时间过长，建议优化",
                current_value=f"平均响应时间: {self.performance_baseline['average_response_time']:.2f}s",
                suggested_value="减少不必要的工具调用，优化决策流程",
                expected_improvement=0.2,
                confidence=0.7,
                supporting_evidence=[
                    f"当前平均响应时间: {self.performance_baseline['average_response_time']:.2f}s",
                    "超过目标值3秒"
                ],
                implementation_steps=[
                    "1. 评估工具调用的必要性",
                    "2. 并行化独立操作",
                    "3. 缓存常用结果"
                ],
                risks=["可能影响回答质量"]
            )
            suggestions.append(suggestion)

        return suggestions

    def _generate_tool_optimization_suggestions(self) -> List[OptimizationSuggestion]:
        """生成工具优化建议"""
        import uuid
        suggestions = []

        # 分析工具使用模式
        tool_stats = defaultdict(lambda: {"uses": 0, "successes": 0, "avg_time": 0.0})

        for profile in self.strategy_profiles.values():
            if "tool" in profile.category.lower():
                tool_stats[profile.strategy_name] = {
                    "uses": profile.total_uses,
                    "successes": profile.success_count,
                    "avg_time": profile.average_execution_time
                }

        # 找出低效工具
        for tool_name, stats in tool_stats.items():
            if stats["uses"] > 10:
                success_rate = stats["successes"] / stats["uses"]

                if success_rate < 0.6:
                    suggestion = OptimizationSuggestion(
                        suggestion_id=f"opt_{uuid.uuid4().hex[:8]}",
                        optimization_type=OptimizationType.TOOL_SELECTION,
                        priority=OptimizationPriority.MEDIUM,
                        description=f"工具 '{tool_name}' 成功率较低",
                        current_value=f"成功率: {success_rate:.2%}",
                        suggested_value="考虑替换为更可靠的工具或增加错误处理",
                        expected_improvement=0.15,
                        confidence=0.6,
                        supporting_evidence=[
                            f"使用次数: {stats['uses']}",
                            f"成功率: {success_rate:.2%}",
                            f"平均执行时间: {stats['avg_time']:.2f}s"
                        ],
                        implementation_steps=[
                            "1. 分析工具失败原因",
                            "2. 寻找替代工具",
                            "3. 增加重试机制",
                            "4. 优化调用参数"
                        ],
                        risks=["可能需要开发新工具"]
                    )
                    suggestions.append(suggestion)

        return suggestions

    def _generate_threshold_adjustments(self) -> List[OptimizationSuggestion]:
        """生成阈值调整建议"""
        import uuid
        suggestions = []

        # 基于风险阈值的建议
        high_risk_failures = sum(
            1 for p in self.strategy_profiles.values()
            if "risk" in p.category.lower() and p.success_rate < 0.5
        )

        if high_risk_failures > 3:
            suggestion = OptimizationSuggestion(
                suggestion_id=f"opt_{uuid.uuid4().hex[:8]}",
                optimization_type=OptimizationType.THRESHOLD_ADJUSTMENT,
                priority=OptimizationPriority.HIGH,
                description="高风险操作失败率高，建议提高风险阈值",
                current_value="当前风险阈值可能过低",
                suggested_value="将风险阈值从0.7提高到0.75-0.8",
                expected_improvement=0.1,
                confidence=0.65,
                supporting_evidence=[
                    f"{high_risk_failures}个高风险策略成功率低于50%",
                    "需要更严格的风险控制"
                ],
                implementation_steps=[
                    "1. 调整SystemConfig.RISK_THRESHOLD",
                    "2. 增强风险评估逻辑",
                    "3. 增加人工确认环节"
                ],
                risks=["可能过度谨慎", "降低自主性"]
            )
            suggestions.append(suggestion)

        return suggestions

    def _update_scenario_info(
        self,
        profile: StrategyProfile,
        context: Dict[str, Any],
        success: bool
    ):
        """更新场景信息"""
        # 提取关键上下文特征
        scenario_features = []

        if "user_intent" in context:
            intent = context["user_intent"]
            if isinstance(intent, dict):
                intent_type = intent.get("intent_type", "unknown")
                scenario_features.append(f"intent:{intent_type}")

        if "complexity" in context:
            complexity = context["complexity"]
            if isinstance(complexity, (int, float)):
                if complexity > 0.7:
                    scenario_features.append("high_complexity")
                elif complexity < 0.3:
                    scenario_features.append("low_complexity")

        # 更新最佳/最差条件
        if success and scenario_features:
            for feature in scenario_features:
                if feature not in profile.best_conditions:
                    profile.best_conditions[feature] = 0
                profile.best_conditions[feature] += 1

        if not success and scenario_features:
            for feature in scenario_features:
                if feature not in profile.worst_conditions:
                    profile.worst_conditions[feature] = 0
                profile.worst_conditions[feature] += 1

    def _update_performance_baseline(self):
        """更新性能基线"""
        if not self.strategy_profiles:
            return

        total_uses = sum(p.total_uses for p in self.strategy_profiles.values())
        if total_uses == 0:
            return

        # 计算加权平均
        weighted_success_rate = sum(
            p.success_rate * p.total_uses
            for p in self.strategy_profiles.values()
        ) / total_uses

        weighted_reward = sum(
            p.average_reward * p.total_uses
            for p in self.strategy_profiles.values()
        ) / total_uses

        weighted_time = sum(
            p.average_execution_time * p.total_uses
            for p in self.strategy_profiles.values()
        ) / total_uses

        self.performance_baseline = {
            "average_success_rate": weighted_success_rate,
            "average_reward": weighted_reward,
            "average_response_time": weighted_time
        }

    def _get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.strategy_profiles:
            return {"message": "无策略数据"}

        # 按类别分组
        category_stats = defaultdict(lambda: {
            "count": 0,
            "total_uses": 0,
            "avg_success_rate": 0.0
        })

        for profile in self.strategy_profiles.values():
            cat = profile.category
            category_stats[cat]["count"] += 1
            category_stats[cat]["total_uses"] += profile.total_uses

            if profile.total_uses > 0:
                current_avg = category_stats[cat]["avg_success_rate"]
                n = category_stats[cat]["count"]
                category_stats[cat]["avg_success_rate"] = (
                    (current_avg * (n - 1) + profile.success_rate) / n
                )

        return {
            "baseline": self.performance_baseline,
            "category_breakdown": dict(category_stats),
            "top_performing_strategies": self._get_top_strategies(5),
            "needs_improvement": self._get_needs_improvement_strategies(5)
        }

    def _get_top_strategies(self, n: int = 5) -> List[Dict[str, Any]]:
        """获取表现最好的策略"""
        sorted_strategies = sorted(
            self.strategy_profiles.values(),
            key=lambda p: p.success_rate * p.total_uses,
            reverse=True
        )

        return [p.to_dict() for p in sorted_strategies[:n]]

    def _get_needs_improvement_strategies(self, n: int = 5) -> List[Dict[str, Any]]:
        """获取需要改进的策略"""
        # 过滤掉使用次数太少的
        qualified = [
            p for p in self.strategy_profiles.values()
            if p.total_uses >= 10
        ]

        sorted_strategies = sorted(
            qualified,
            key=lambda p: p.success_rate
        )

        return [p.to_dict() for p in sorted_strategies[:n]]

    def _save_to_disk(self):
        """保存到磁盘"""
        try:
            filepath = os.path.join(
                self.config.storage_path,
                f"optimizer_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            data = {
                "strategy_profiles": {
                    name: profile.to_dict()
                    for name, profile in self.strategy_profiles.items()
                },
                "stats": self.stats,
                "performance_baseline": self.performance_baseline,
                "optimization_history": self.optimization_history[-20:],  # 只保留最近20条
                "timestamp": datetime.now().isoformat()
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.logger.debug(f"💾 优化器状态已保存: {filepath}")

            # 清理旧文件
            self._cleanup_old_files()

        except Exception as e:
            self.logger.error(f"❌ 保存优化器状态失败: {str(e)}")

    def _load_from_disk(self):
        """从磁盘加载"""
        try:
            if not os.path.exists(self.config.storage_path):
                return

            files = [
                f for f in os.listdir(self.config.storage_path)
                if f.startswith("optimizer_state_") and f.endswith(".json")
            ]

            if not files:
                return

            latest_file = sorted(files)[-1]
            filepath = os.path.join(self.config.storage_path, latest_file)

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 恢复策略档案（简化版，实际应该完整恢复）
            profiles_data = data.get("strategy_profiles", {})
            for name, profile_data in profiles_data.items():
                profile = StrategyProfile(
                    strategy_name=profile_data["strategy_name"],
                    category=profile_data["category"]
                )
                profile.total_uses = profile_data.get("total_uses", 0)
                profile.success_count = profile_data.get("success_count", 0)
                profile.failure_count = profile_data.get("failure_count", 0)
                profile.average_reward = profile_data.get("average_reward", 0.0)
                profile.average_execution_time = profile_data.get("average_execution_time", 0.0)

                self.strategy_profiles[name] = profile

            # 恢复统计
            if "stats" in data:
                self.stats.update(data["stats"])

            if "performance_baseline" in data:
                self.performance_baseline.update(data["performance_baseline"])

            self.logger.info(f"📂 已加载优化器状态 - {len(self.strategy_profiles)} 个策略")

        except Exception as e:
            self.logger.error(f"❌ 加载优化器状态失败: {str(e)}")

    def _cleanup_old_files(self, keep_count: int = 5):
        """清理旧文件"""
        try:
            files = [
                f for f in os.listdir(self.config.storage_path)
                if f.startswith("optimizer_state_") and f.endswith(".json")
            ]

            if len(files) > keep_count:
                sorted_files = sorted(files)
                for old_file in sorted_files[:-keep_count]:
                    filepath = os.path.join(self.config.storage_path, old_file)
                    os.remove(filepath)
                    self.logger.debug(f"🗑️ 已删除旧文件: {old_file}")

        except Exception as e:
            self.logger.error(f"❌ 清理旧文件失败: {str(e)}")
