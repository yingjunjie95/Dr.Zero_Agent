"""
持续学习器 - Continuous Learner
实现Agent的在线学习和自适应能力
通过记录交互经验、调整学习参数和更新策略，持续提升Agent性能
专为CPU环境优化，提供轻量级的增量学习机制
"""
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque


@dataclass
class LearningMetrics:
    """学习指标"""
    total_interactions: int = 0
    successful_interactions: int = 0
    failed_interactions: int = 0
    average_reward: float = 0.0
    learning_progress: float = 0.0

    # 性能趋势
    recent_rewards: deque = field(default_factory=lambda: deque(maxlen=50))
    improvement_trend: str = "stable"  # improving, declining, stable

    def update(self, reward: float, success: bool):
        """更新学习指标"""
        self.total_interactions += 1

        if success:
            self.successful_interactions += 1
        else:
            self.failed_interactions += 1

        # 更新平均奖励（移动平均）
        if self.total_interactions == 1:
            self.average_reward = reward
        else:
            self.average_reward = (
                (self.average_reward * (self.total_interactions - 1) + reward)
                / self.total_interactions
            )

        # 记录最近奖励
        self.recent_rewards.append(reward)

        # 更新趋势
        self._update_trend()

    def _update_trend(self):
        """更新改进趋势"""
        if len(self.recent_rewards) < 10:
            self.improvement_trend = "insufficient_data"
            return

        recent = list(self.recent_rewards)
        mid = len(recent) // 2

        first_half_avg = sum(recent[:mid]) / mid
        second_half_avg = sum(recent[mid:]) / (len(recent) - mid)

        change = second_half_avg - first_half_avg

        if change > 0.1:
            self.improvement_trend = "improving"
        elif change < -0.1:
            self.improvement_trend = "declining"
        else:
            self.improvement_trend = "stable"

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_interactions == 0:
            return 0.0
        return self.successful_interactions / self.total_interactions

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_interactions": self.total_interactions,
            "success_rate": round(self.success_rate, 3),
            "average_reward": round(self.average_reward, 3),
            "learning_progress": round(self.learning_progress, 3),
            "improvement_trend": self.improvement_trend,
            "recent_performance": {
                "last_10_avg": (
                    round(sum(list(self.recent_rewards)[-10:]) / min(10, len(self.recent_rewards)), 3)
                    if self.recent_rewards else 0.0
                )
            }
        }


class ContinuousLearner:
    """
    持续学习器

    功能：
    - 交互记录：记录每次交互的学习数据
    - 参数调整：动态调整学习和探索参数
    - 性能跟踪：监控学习进度和效果
    - 知识积累：从成功和失败中提取模式
    - 自适应学习：根据表现自动调整学习策略
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        exploration_rate: float = 0.2,
        min_learning_rate: float = 0.01,
        max_learning_rate: float = 0.5,
        decay_factor: float = 0.995
    ):
        """
        初始化持续学习器

        Args:
            learning_rate: 学习率
            exploration_rate: 探索率
            min_learning_rate: 最小学习率
            max_learning_rate: 最大学习率
            decay_factor: 衰减因子
        """
        self.base_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.decay_factor = decay_factor

        self.logger = logging.getLogger("ContinuousLearner")

        # 学习指标
        self.metrics = LearningMetrics()

        # 学习历史
        self.learning_history: List[Dict[str, Any]] = []
        self.max_history_size = 200

        # 模式库
        self.success_patterns: List[Dict[str, Any]] = []
        self.failure_patterns: List[Dict[str, Any]] = []

        # 参数调整历史
        self.parameter_adjustments: List[Dict[str, Any]] = []

        self.logger.info(
            f"✅ 持续学习器初始化完成 - "
            f"学习率: {learning_rate}, 探索率: {exploration_rate}"
        )

    def record_interaction(self, interaction_data: Dict[str, Any]):
        """
        记录交互数据用于学习

        Args:
            interaction_data: 交互数据字典
        """
        try:
            # 提取关键信息
            outcome = interaction_data.get('outcome', 'unknown')
            success = outcome == 'success'

            # 计算奖励
            reward = self._calculate_reward(interaction_data)

            # 更新指标
            self.metrics.update(reward, success)

            # 记录到历史
            learning_record = {
                "timestamp": datetime.now().isoformat(),
                "outcome": outcome,
                "reward": reward,
                "success": success,
                "learning_rate": self.current_learning_rate,
                "exploration_rate": self.exploration_rate,
                "interaction_summary": self._summarize_interaction(interaction_data)
            }

            self.learning_history.append(learning_record)

            # 限制历史记录大小
            if len(self.learning_history) > self.max_history_size:
                self.learning_history.pop(0)

            # 记录模式
            if success:
                self._record_success_pattern(interaction_data, reward)
            else:
                self._record_failure_pattern(interaction_data, reward)

            # 定期调整参数
            if self.metrics.total_interactions % 10 == 0:
                self._adjust_learning_parameters()

            self.logger.debug(
                f"📚 交互已记录 - 结果: {outcome}, "
                f"奖励: {reward:.2f}, 学习率: {self.current_learning_rate:.3f}"
            )

        except Exception as e:
            self.logger.error(f"❌ 记录交互失败: {str(e)}", exc_info=True)

    def get_learning_state(self) -> Dict[str, Any]:
        """
        获取当前学习状态

        Returns:
            学习状态字典
        """
        return {
            "metrics": self.metrics.to_dict(),
            "current_parameters": {
                "learning_rate": self.current_learning_rate,
                "exploration_rate": self.exploration_rate
            },
            "pattern_counts": {
                "success_patterns": len(self.success_patterns),
                "failure_patterns": len(self.failure_patterns)
            },
            "total_records": len(self.learning_history)
        }

    def should_explore(self) -> bool:
        """
        判断是否应该探索新策略

        Returns:
            是否应该探索
        """
        import random
        return random.random() < self.exploration_rate

    def get_learned_patterns(self, pattern_type: str = "success") -> List[Dict[str, Any]]:
        """
        获取已学习的模式

        Args:
            pattern_type: 模式类型 (success/failure)

        Returns:
            模式列表
        """
        if pattern_type == "success":
            return self.success_patterns[-10:]
        elif pattern_type == "failure":
            return self.failure_patterns[-10:]
        else:
            return []

    def _calculate_reward(self, interaction_data: Dict[str, Any]) -> float:
        """
        计算奖励信号

        Args:
            interaction_data: 交互数据

        Returns:
            奖励值 (-1.0 to 1.0)
        """
        outcome = interaction_data.get('outcome', 'unknown')
        execution_metrics = interaction_data.get('execution_metrics', {})
        cognitive_state = interaction_data.get('cognitive_state', {})

        # 基础奖励
        if outcome == 'success':
            reward = 0.5
        elif outcome == 'failure':
            reward = -0.5
        else:
            reward = 0.0

        # 执行效率奖励
        errors = execution_metrics.get('errors', 0)
        if errors == 0:
            reward += 0.2
        else:
            reward -= 0.1 * errors

        # 置信度奖励（高置信度且成功给予额外奖励）
        confidence = cognitive_state.get('confidence_score', 0.5)
        if outcome == 'success' and confidence > 0.7:
            reward += 0.1
        elif outcome == 'failure' and confidence > 0.7:
            reward -= 0.2  # 高置信度但失败，惩罚更大

        # 执行时间惩罚（过长则惩罚）
        execution_time = execution_metrics.get('execution_time', 0.0)
        if execution_time > 10.0:
            reward -= 0.1

        # 限制在 [-1.0, 1.0] 范围
        return max(min(reward, 1.0), -1.0)

    def _summarize_interaction(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成交互摘要"""
        decision = interaction_data.get('decision', {})
        perception = interaction_data.get('perception', {})

        return {
            "primary_action": decision.get('primary_action', 'unknown'),
            "strategy_type": decision.get('strategy_type', 'unknown'),
            "user_input_length": len(perception.get('user_input', '')),
            "tool_calls": decision.get('tool_calls', [])
        }

    def _record_success_pattern(
        self,
        interaction_data: Dict[str, Any],
        reward: float
    ):
        """记录成功模式"""
        decision = interaction_data.get('decision', {})
        cognitive_state = interaction_data.get('cognitive_state', {})

        pattern = {
            "timestamp": datetime.now().isoformat(),
            "strategy": decision.get('strategy_type', 'unknown'),
            "action": decision.get('primary_action', 'unknown'),
            "confidence": cognitive_state.get('confidence_score', 0.0),
            "reward": reward,
            "context_summary": self._summarize_interaction(interaction_data)
        }

        self.success_patterns.append(pattern)

        # 限制模式数量
        if len(self.success_patterns) > 50:
            self.success_patterns.pop(0)

    def _record_failure_pattern(
        self,
        interaction_data: Dict[str, Any],
        reward: float
    ):
        """记录失败模式"""
        decision = interaction_data.get('decision', {})
        cognitive_state = interaction_data.get('cognitive_state', {})
        execution_metrics = interaction_data.get('execution_metrics', {})

        pattern = {
            "timestamp": datetime.now().isoformat(),
            "strategy": decision.get('strategy_type', 'unknown'),
            "action": decision.get('primary_action', 'unknown'),
            "confidence": cognitive_state.get('confidence_score', 0.0),
            "reward": reward,
            "errors": execution_metrics.get('errors', 0),
            "context_summary": self._summarize_interaction(interaction_data)
        }

        self.failure_patterns.append(pattern)

        # 限制模式数量
        if len(self.failure_patterns) > 50:
            self.failure_patterns.pop(0)

    def _adjust_learning_parameters(self):
        """调整学习参数"""
        old_learning_rate = self.current_learning_rate
        old_exploration_rate = self.exploration_rate

        # 基于性能调整学习率
        if self.metrics.improvement_trend == "improving":
            # 性能提升，保持或略微降低学习率
            self.current_learning_rate *= 0.98
        elif self.metrics.improvement_trend == "declining":
            # 性能下降，提高学习率以更快适应
            self.current_learning_rate *= 1.05
        else:
            # 稳定，缓慢衰减
            self.current_learning_rate *= self.decay_factor

        # 限制学习率范围
        self.current_learning_rate = max(
            self.min_learning_rate,
            min(self.max_learning_rate, self.current_learning_rate)
        )

        # 基于成功率调整探索率
        success_rate = self.metrics.success_rate
        if success_rate > 0.8:
            # 成功率高，减少探索
            self.exploration_rate *= 0.95
        elif success_rate < 0.5:
            # 成功率低，增加探索
            self.exploration_rate *= 1.05

        # 限制探索率范围
        self.exploration_rate = max(0.05, min(0.5, self.exploration_rate))

        # 记录参数调整
        adjustment = {
            "timestamp": datetime.now().isoformat(),
            "learning_rate": {
                "old": old_learning_rate,
                "new": self.current_learning_rate
            },
            "exploration_rate": {
                "old": old_exploration_rate,
                "new": self.exploration_rate
            },
            "reason": f"trend={self.metrics.improvement_trend}, success_rate={success_rate:.2f}"
        }

        self.parameter_adjustments.append(adjustment)

        self.logger.info(
            f"🎯 学习参数已调整 - "
            f"学习率: {old_learning_rate:.3f} → {self.current_learning_rate:.3f}, "
            f"探索率: {old_exploration_rate:.3f} → {self.exploration_rate:.3f}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "metrics": self.metrics.to_dict(),
            "parameters": {
                "base_learning_rate": self.base_learning_rate,
                "current_learning_rate": self.current_learning_rate,
                "exploration_rate": self.exploration_rate,
                "min_learning_rate": self.min_learning_rate,
                "max_learning_rate": self.max_learning_rate,
                "decay_factor": self.decay_factor
            },
            "patterns": {
                "success_count": len(self.success_patterns),
                "failure_count": len(self.failure_patterns)
            },
            "history_size": len(self.learning_history),
            "parameter_adjustments_count": len(self.parameter_adjustments)
        }
