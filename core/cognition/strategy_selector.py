"""
策略选择器 - Strategy Selector
根据元认知评估结果、可用工具和资源约束，动态选择最优执行策略
实现智能化的策略匹配和自适应决策
专为CPU环境优化，提供高效的策略选择和优先级排序
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class StrategyType(Enum):
    """策略类型"""
    DIRECT_ANSWER = "direct_answer"  # 直接回答
    TOOL_ASSISTED = "tool_assisted"  # 工具辅助
    CLARIFICATION = "clarification"  # 澄清询问
    DECOMPOSITION = "decomposition"  # 任务分解
    ESCALATION = "escalation"  # 升级处理
    DEFERRED = "deferred"  # 延迟处理
    EXPLORATORY = "exploratory"  # 探索性处理


class StrategyPriority(Enum):
    """策略优先级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Strategy:
    """策略定义"""
    strategy_id: str
    strategy_type: StrategyType
    name: str
    description: str
    priority: StrategyPriority = StrategyPriority.MEDIUM

    # 适用条件
    min_confidence: float = 0.0  # 最小置信度要求
    max_complexity: float = 1.0  # 最大复杂度限制
    required_tools: List[str] = field(default_factory=list)  # 需要的工具
    risk_tolerance: float = 1.0  # 风险容忍度 (0.0-1.0)

    # 执行参数
    primary_action: str = "direct_response"
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    memory_operations: List[Dict[str, Any]] = field(default_factory=list)
    fallback_actions: List[str] = field(default_factory=list)
    requires_confirmation: bool = False
    max_execution_time: float = 30.0
    confidence_threshold: float = 0.6

    # 性能统计
    usage_count: int = 0
    success_rate: float = 0.0
    average_execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "strategy_id": self.strategy_id,
            "strategy_type": self.strategy_type.value,
            "name": self.name,
            "description": self.description,
            "priority": self.priority.value,
            "min_confidence": self.min_confidence,
            "max_complexity": self.max_complexity,
            "required_tools": self.required_tools,
            "risk_tolerance": self.risk_tolerance,
            "primary_action": self.primary_action,
            "tool_calls": self.tool_calls,
            "memory_operations": self.memory_operations,
            "fallback_actions": self.fallback_actions,
            "requires_confirmation": self.requires_confirmation,
            "max_execution_time": self.max_execution_time,
            "confidence_threshold": self.confidence_threshold,
            "usage_count": self.usage_count,
            "success_rate": round(self.success_rate, 3),
            "average_execution_time": round(self.average_execution_time, 3)
        }


class StrategySelector:
    """
    策略选择器

    功能：
    - 策略库管理：维护和管理可用策略集合
    - 条件匹配：根据情境评估结果匹配合适策略
    - 优先级排序：对候选策略进行评分和排序
    - 动态调整：基于历史表现调整策略选择
    - 回退机制：在主策略不可用时提供备选方案
    """

    def __init__(self):
        """初始化策略选择器"""
        self.logger = logging.getLogger("StrategySelector")

        # 策略库
        self.strategy_library: Dict[str, Strategy] = {}

        # 初始化默认策略
        self._initialize_default_strategies()

        # 选择历史
        self.selection_history: List[Dict[str, Any]] = []
        self.max_history_size = 100

        # 统计信息
        self.stats = {
            "total_selections": 0,
            "strategy_usage": {},
            "average_confidence": 0.0
        }

        self.logger.info(f"✅ 策略选择器初始化完成 - {len(self.strategy_library)} 个策略")

    def select_strategy(
        self,
        metacognitive_assessment: Dict[str, Any],
        available_tools: List[str],
        resource_constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        选择最优策略

        Args:
            metacognitive_assessment: 元认知评估结果
            available_tools: 可用工具列表
            resource_constraints: 资源约束

        Returns:
            选中的策略字典
        """
        try:
            self.logger.debug("🎯 开始策略选择...")

            # 1. 提取关键评估指标
            confidence = metacognitive_assessment.get('overall_confidence', 0.5)
            complexity = metacognitive_assessment.get('task_complexity', 0.5)
            risk_level = metacognitive_assessment.get('risk_level', 'low')
            requires_clarification = metacognitive_assessment.get('requires_clarification', False)
            has_goal = metacognitive_assessment.get('has_goal', False)

            # 2. 过滤符合条件的策略
            candidate_strategies = self._filter_candidates(
                confidence=confidence,
                complexity=complexity,
                risk_level=risk_level,
                available_tools=available_tools,
                resource_constraints=resource_constraints
            )

            if not candidate_strategies:
                self.logger.warning("⚠️ 没有符合条件的策略，使用默认策略")
                selected_strategy = self._get_default_strategy()
            else:
                # 3. 如果需要澄清，优先选择澄清策略
                if requires_clarification:
                    clarification_strategy = self._find_clarification_strategy(candidate_strategies)
                    if clarification_strategy:
                        selected_strategy = clarification_strategy
                    else:
                        selected_strategy = self._score_and_select(candidate_strategies, confidence)
                else:
                    # 4. 评分并选择最优策略
                    selected_strategy = self._score_and_select(candidate_strategies, confidence)

            # 5. 更新统计
            self._update_statistics(selected_strategy)

            self.logger.debug(
                f"✨ 策略已选择: {selected_strategy.name} "
                f"(类型: {selected_strategy.strategy_type.value})"
            )

            return selected_strategy.to_dict()

        except Exception as e:
            self.logger.error(f"❌ 策略选择失败: {str(e)}", exc_info=True)
            return self._get_fallback_strategy().to_dict()

    def _initialize_default_strategies(self):
        """初始化默认策略库"""
        strategies = [
            # 1. 直接回答策略（高置信度、低复杂度）
            Strategy(
                strategy_id="direct_answer_01",
                strategy_type=StrategyType.DIRECT_ANSWER,
                name="直接回答",
                description="基于已有知识直接回答问题",
                priority=StrategyPriority.HIGH,
                min_confidence=0.7,
                max_complexity=0.4,
                risk_tolerance=0.8,
                primary_action="direct_response",
                fallback_actions=["clarify", "use_tools"],
                max_execution_time=5.0,
                confidence_threshold=0.7
            ),

            # 2. 工具辅助策略（中等置信度、需要工具）
            Strategy(
                strategy_id="tool_assisted_01",
                strategy_type=StrategyType.TOOL_ASSISTED,
                name="工具辅助",
                description="调用外部工具增强处理能力",
                priority=StrategyPriority.HIGH,
                min_confidence=0.5,
                max_complexity=0.7,
                risk_tolerance=0.6,
                primary_action="use_tools",
                fallback_actions=["clarify", "escalate"],
                requires_confirmation=False,
                max_execution_time=30.0,
                confidence_threshold=0.6
            ),

            # 3. 澄清询问策略（低置信度或需要澄清）
            Strategy(
                strategy_id="clarification_01",
                strategy_type=StrategyType.CLARIFICATION,
                name="澄清询问",
                description="向用户询问更多信息以明确需求",
                priority=StrategyPriority.MEDIUM,
                min_confidence=0.0,
                max_complexity=1.0,
                risk_tolerance=1.0,
                primary_action="clarify",
                fallback_actions=["escalate"],
                requires_confirmation=False,
                max_execution_time=3.0,
                confidence_threshold=0.3
            ),

            # 4. 任务分解策略（高复杂度）
            Strategy(
                strategy_id="decomposition_01",
                strategy_type=StrategyType.DECOMPOSITION,
                name="任务分解",
                description="将复杂任务分解为多个子任务",
                priority=StrategyPriority.MEDIUM,
                min_confidence=0.6,
                max_complexity=1.0,
                risk_tolerance=0.7,
                primary_action="use_tools",
                fallback_actions=["clarify", "escalate"],
                requires_confirmation=False,
                max_execution_time=60.0,
                confidence_threshold=0.6
            ),

            # 5. 升级处理策略（高风险或超出能力）
            Strategy(
                strategy_id="escalation_01",
                strategy_type=StrategyType.ESCALATION,
                name="升级处理",
                description="超出能力范围，建议寻求其他帮助",
                priority=StrategyPriority.LOW,
                min_confidence=0.0,
                max_complexity=1.0,
                risk_tolerance=0.3,
                primary_action="escalate",
                fallback_actions=[],
                requires_confirmation=False,
                max_execution_time=2.0,
                confidence_threshold=0.2
            ),

            # 6. 探索性处理策略（中等置信度、创造性任务）
            Strategy(
                strategy_id="exploratory_01",
                strategy_type=StrategyType.EXPLORATORY,
                name="探索性处理",
                description="尝试多种方法探索解决方案",
                priority=StrategyPriority.MEDIUM,
                min_confidence=0.5,
                max_complexity=0.8,
                risk_tolerance=0.7,
                primary_action="use_tools",
                fallback_actions=["clarify", "direct_response"],
                requires_confirmation=False,
                max_execution_time=45.0,
                confidence_threshold=0.5
            )
        ]

        for strategy in strategies:
            self.strategy_library[strategy.strategy_id] = strategy

    def _filter_candidates(
        self,
        confidence: float,
        complexity: float,
        risk_level: str,
        available_tools: List[str],
        resource_constraints: Dict[str, Any]
    ) -> List[Strategy]:
        """过滤符合条件的候选策略"""
        candidates = []

        # 风险等级映射
        risk_map = {
            "low": 0.25,
            "medium": 0.5,
            "high": 0.75,
            "critical": 1.0
        }
        current_risk = risk_map.get(risk_level, 0.5)

        for strategy in self.strategy_library.values():
            # 检查置信度要求
            if confidence < strategy.min_confidence:
                continue

            # 检查复杂度限制
            if complexity > strategy.max_complexity:
                continue

            # 检查风险容忍度
            if current_risk > strategy.risk_tolerance:
                continue

            # 检查工具可用性
            if strategy.required_tools:
                if not all(tool in available_tools for tool in strategy.required_tools):
                    continue

            # 检查资源约束
            if resource_constraints.get('memory_constrained', False):
                if strategy.max_execution_time > 10.0:
                    continue

            if resource_constraints.get('cpu_constrained', False):
                if strategy.strategy_type == StrategyType.DECOMPOSITION:
                    continue

            candidates.append(strategy)

        return candidates

    def _score_and_select(
        self,
        candidates: List[Strategy],
        confidence: float
    ) -> Strategy:
        """评分并选择最优策略"""
        if not candidates:
            return self._get_default_strategy()

        if len(candidates) == 1:
            return candidates[0]

        # 计算每个候选策略的得分
        scored_candidates = []
        for strategy in candidates:
            score = self._calculate_strategy_score(strategy, confidence)
            scored_candidates.append((score, strategy))

        # 按得分排序（降序）
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # 选择得分最高的策略
        best_score, best_strategy = scored_candidates[0]

        self.logger.debug(
            f"📊 策略评分: {best_strategy.name} - 得分: {best_score:.2f}"
        )

        return best_strategy

    def _calculate_strategy_score(
        self,
        strategy: Strategy,
        confidence: float
    ) -> float:
        """计算策略得分"""
        score = 0.0

        # 1. 优先级得分 (0-40分)
        priority_scores = {
            StrategyPriority.LOW: 10,
            StrategyPriority.MEDIUM: 20,
            StrategyPriority.HIGH: 30,
            StrategyPriority.CRITICAL: 40
        }
        score += priority_scores.get(strategy.priority, 20)

        # 2. 置信度匹配得分 (0-30分)
        # 置信度越高，越倾向于选择要求高的策略
        confidence_match = 1.0 - abs(confidence - strategy.min_confidence)
        score += confidence_match * 30

        # 3. 历史成功率得分 (0-20分)
        score += strategy.success_rate * 20

        # 4. 执行效率得分 (0-10分)
        # 执行时间越短，得分越高
        efficiency = max(0, 1.0 - strategy.max_execution_time / 60.0)
        score += efficiency * 10

        return score

    def _find_clarification_strategy(
        self,
        candidates: List[Strategy]
    ) -> Optional[Strategy]:
        """查找澄清策略"""
        for strategy in candidates:
            if strategy.strategy_type == StrategyType.CLARIFICATION:
                return strategy
        return None

    def _get_default_strategy(self) -> Strategy:
        """获取默认策略"""
        # 优先返回直接回答策略
        for strategy in self.strategy_library.values():
            if strategy.strategy_type == StrategyType.DIRECT_ANSWER:
                return strategy

        # 如果没有，返回第一个策略
        return next(iter(self.strategy_library.values()))

    def _get_fallback_strategy(self) -> Strategy:
        """获取回退策略"""
        # 回退到澄清策略
        for strategy in self.strategy_library.values():
            if strategy.strategy_type == StrategyType.CLARIFICATION:
                return strategy

        return self._get_default_strategy()

    def _update_statistics(self, selected_strategy: Strategy):
        """更新统计信息"""
        self.stats["total_selections"] += 1

        strategy_type = selected_strategy.strategy_type.value
        if strategy_type not in self.stats["strategy_usage"]:
            self.stats["strategy_usage"][strategy_type] = 0
        self.stats["strategy_usage"][strategy_type] += 1

        # 更新策略使用计数
        selected_strategy.usage_count += 1

        # 记录选择历史
        self.selection_history.append({
            "timestamp": datetime.now(),
            "strategy_id": selected_strategy.strategy_id,
            "strategy_type": strategy_type
        })

        # 限制历史记录大小
        if len(self.selection_history) > self.max_history_size:
            self.selection_history.pop(0)

    def update_strategy_performance(
        self,
        strategy_id: str,
        success: bool,
        execution_time: float
    ):
        """
        更新策略性能统计

        Args:
            strategy_id: 策略ID
            success: 是否成功
            execution_time: 执行时间
        """
        if strategy_id not in self.strategy_library:
            return

        strategy = self.strategy_library[strategy_id]

        # 更新成功率（移动平均）
        if strategy.usage_count == 1:
            strategy.success_rate = 1.0 if success else 0.0
        else:
            old_success_rate = strategy.success_rate
            strategy.success_rate = (
                old_success_rate * (strategy.usage_count - 1) +
                (1.0 if success else 0.0)
            ) / strategy.usage_count

        # 更新平均执行时间
        if strategy.usage_count == 1:
            strategy.average_execution_time = execution_time
        else:
            old_avg_time = strategy.average_execution_time
            strategy.average_execution_time = (
                old_avg_time * (strategy.usage_count - 1) +
                execution_time
            ) / strategy.usage_count

        self.logger.debug(
            f"📈 策略性能已更新: {strategy_id} - "
            f"成功率: {strategy.success_rate:.2%}, "
            f"平均时间: {strategy.average_execution_time:.2f}s"
        )

    def get_strategy_statistics(self) -> Dict[str, Any]:
        """获取策略统计信息"""
        return {
            **self.stats,
            "total_strategies": len(self.strategy_library),
            "strategy_details": {
                sid: {
                    "name": s.name,
                    "type": s.strategy_type.value,
                    "usage_count": s.usage_count,
                    "success_rate": round(s.success_rate, 3),
                    "average_execution_time": round(s.average_execution_time, 3)
                }
                for sid, s in self.strategy_library.items()
            }
        }

    def get_available_strategies(self) -> List[Dict[str, Any]]:
        """获取所有可用策略"""
        return [strategy.to_dict() for strategy in self.strategy_library.values()]
