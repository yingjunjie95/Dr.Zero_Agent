"""
自我模型 - Self Model
Agent的自我认知表示，维护对自身能力、知识、局限性和历史表现的理解
这是元认知的基础，使Agent能够进行自我评估和反思
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from enum import Enum


class CapabilityDomain(Enum):
    """能力领域"""
    LANGUAGE_UNDERSTANDING = "language_understanding"
    LOGICAL_REASONING = "logical_reasoning"
    CREATIVE_THINKING = "creative_thinking"
    ANALYTICAL_DEPTH = "analytical_depth"
    EMOTIONAL_INTELLIGENCE = "emotional_intelligence"
    MEMORY_RECALL = "memory_recall"
    TOOL_USAGE = "tool_usage"
    ADAPTABILITY = "adaptability"
    CODE_GENERATION = "code_generation"
    DATA_ANALYSIS = "data_analysis"
    PROBLEM_SOLVING = "problem_solving"


class PerformanceTrend(Enum):
    """性能趋势"""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"


@dataclass
class CapabilityProfile:
    """能力档案"""
    domain: CapabilityDomain
    proficiency: float  # 熟练度 (0.0-1.0)
    confidence: float  # 对该能力的自信程度 (0.0-1.0)
    experience_count: int  # 使用该能力的次数
    last_used: datetime  # 最后使用时间
    success_rate: float  # 成功率
    avg_performance: float  # 平均表现

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain.value,
            "proficiency": round(self.proficiency, 3),
            "confidence": round(self.confidence, 3),
            "experience_count": self.experience_count,
            "last_used": self.last_used.isoformat(),
            "success_rate": round(self.success_rate, 3),
            "avg_performance": round(self.avg_performance, 3)
        }


@dataclass
class PerformanceRecord:
    """性能记录"""
    timestamp: datetime
    task_type: str
    complexity: float
    success: bool
    response_time: float
    confidence_before: float  # 任务前的自信度
    confidence_after: float  # 任务后的自信度（基于结果调整）
    user_feedback: Optional[float] = None  # 用户反馈 (0.0-1.0)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "task_type": self.task_type,
            "complexity": round(self.complexity, 3),
            "success": self.success,
            "response_time": round(self.response_time, 3),
            "confidence_before": round(self.confidence_before, 3),
            "confidence_after": round(self.confidence_after, 3),
            "user_feedback": round(self.user_feedback, 3) if self.user_feedback else None,
            "notes": self.notes
        }


@dataclass
class KnowledgeBoundary:
    """知识边界"""
    domain: str
    expertise_level: float  # 专业水平 (0.0-1.0)
    is_allowed: bool  # 是否允许
    is_forbidden: bool  # 是否禁止
    last_updated: datetime = field(default_factory=datetime.now)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "expertise_level": round(self.expertise_level, 3),
            "is_allowed": self.is_allowed,
            "is_forbidden": self.is_forbidden,
            "last_updated": self.last_updated.isoformat(),
            "notes": self.notes
        }


class SelfModel:
    """
    自我模型 - Agent的自我认知核心

    功能：
    - 能力建模：维护和更新各项能力的熟练度
    - 知识管理：跟踪知识边界和专业领域
    - 性能追踪：记录和分析历史表现
    - 自我反思：定期评估和调整自我认知
    - 趋势分析：识别性能变化模式
    """

    def __init__(self, agent_profile: Any):
        """
        初始化自我模型

        Args:
            agent_profile: Agent配置文件
        """
        self.profile = agent_profile
        self.logger = logging.getLogger("SelfModel")

        # 能力模型
        self.capabilities: Dict[CapabilityDomain, CapabilityProfile] = {}
        self._initialize_capabilities()

        # 知识边界
        self.knowledge_boundaries: Dict[str, KnowledgeBoundary] = {}
        self._initialize_knowledge_boundaries()

        # 性能基线
        self.performance_baseline = {
            "avg_response_time": 2.0,
            "success_rate": 0.95,
            "user_satisfaction": 0.85,
            "avg_confidence": 0.75
        }

        # 历史记录
        self.performance_history: deque = deque(maxlen=100)
        self.reflection_history: deque = deque(maxlen=50)

        # 统计信息
        self.total_interactions = 0
        self.total_successes = 0
        self.total_failures = 0

        # 自我反思
        self.last_self_reflection = datetime.now()
        self.reflection_count = 0
        self.self_awareness_score = 0.7  # 初始自我意识得分

        # 学习参数
        self.learning_momentum = 1.0  # 学习动量
        self.adaptation_rate = agent_profile.LEARNING_RATE

        self.logger.info("✅ 自我模型初始化完成")
        self.logger.info(f"   - 能力维度: {len(self.capabilities)}")
        self.logger.info(f"   - 知识领域: {len(self.knowledge_boundaries)}")

    def _initialize_capabilities(self):
        """初始化能力模型"""
        now = datetime.now()

        capability_configs = {
            CapabilityDomain.LANGUAGE_UNDERSTANDING: {
                "base_proficiency": 0.9,
                "confidence": 0.92
            },
            CapabilityDomain.LOGICAL_REASONING: {
                "base_proficiency": 0.85,
                "confidence": 0.87
            },
            CapabilityDomain.CREATIVE_THINKING: {
                "base_proficiency": self.profile.CREATIVITY_LEVEL,
                "confidence": 0.75
            },
            CapabilityDomain.ANALYTICAL_DEPTH: {
                "base_proficiency": self.profile.ANALYTICAL_DEPTH,
                "confidence": 0.80
            },
            CapabilityDomain.EMOTIONAL_INTELLIGENCE: {
                "base_proficiency": self.profile.EMPATHY_LEVEL,
                "confidence": 0.78
            },
            CapabilityDomain.MEMORY_RECALL: {
                "base_proficiency": 0.88,
                "confidence": 0.85
            },
            CapabilityDomain.TOOL_USAGE: {
                "base_proficiency": 0.82,
                "confidence": 0.80
            },
            CapabilityDomain.ADAPTABILITY: {
                "base_proficiency": 0.75,
                "confidence": 0.72
            },
            CapabilityDomain.CODE_GENERATION: {
                "base_proficiency": 0.85,
                "confidence": 0.83
            },
            CapabilityDomain.DATA_ANALYSIS: {
                "base_proficiency": 0.80,
                "confidence": 0.78
            },
            CapabilityDomain.PROBLEM_SOLVING: {
                "base_proficiency": 0.82,
                "confidence": 0.80
            }
        }

        for domain, config in capability_configs.items():
            self.capabilities[domain] = CapabilityProfile(
                domain=domain,
                proficiency=config["base_proficiency"],
                confidence=config["confidence"],
                experience_count=0,
                last_used=now,
                success_rate=0.9,
                avg_performance=0.85
            )

    def _initialize_knowledge_boundaries(self):
        """初始化知识边界"""
        # 允许的领域
        for domain in self.profile.ALLOWED_DOMAINS:
            self.knowledge_boundaries[domain] = KnowledgeBoundary(
                domain=domain,
                expertise_level=0.8,
                is_allowed=True,
                is_forbidden=False
            )

        # 禁止的领域
        for domain in self.profile.FORBIDDEN_DOMAINS:
            self.knowledge_boundaries[domain] = KnowledgeBoundary(
                domain=domain,
                expertise_level=0.0,
                is_allowed=False,
                is_forbidden=True,
                notes="明确禁止的领域"
            )

    def assess_capability(self, task_type: str, complexity: float) -> float:
        """
        评估处理特定任务的能力

        Args:
            task_type: 任务类型
            complexity: 任务复杂度 (0.0-1.0)

        Returns:
            能力得分 (0.0-1.0)
        """
        # 映射任务类型到能力领域
        capability_domain = self._map_task_to_capability(task_type)

        if capability_domain in self.capabilities:
            cap_profile = self.capabilities[capability_domain]
            base_capability = cap_profile.proficiency

            # 考虑成功率
            success_factor = cap_profile.success_rate

            # 复杂度衰减
            complexity_factor = max(0.3, 1.0 - complexity * 0.5)

            # 经验加成
            experience_bonus = min(cap_profile.experience_count * 0.001, 0.1)

            # 综合得分
            capability_score = (
                base_capability * 0.4 +
                success_factor * 0.3 +
                complexity_factor * 0.2 +
                experience_bonus * 0.1
            )

            return min(capability_score, 1.0)
        else:
            # 未知能力类型，返回默认值
            return 0.7 * max(0.3, 1.0 - complexity * 0.5)

    def check_domain_permission(self, domain: str) -> Tuple[bool, str]:
        """
        检查领域权限

        Args:
            domain: 任务领域

        Returns:
            (是否允许, 原因说明)
        """
        if domain in self.knowledge_boundaries:
            boundary = self.knowledge_boundaries[domain]

            if boundary.is_forbidden:
                return False, f"领域 '{domain}' 在禁止列表中"

            if not boundary.is_allowed:
                return False, f"领域 '{domain}' 未授权"

            # 检查专业水平
            if boundary.expertise_level < 0.3:
                return False, f"领域 '{domain}' 专业水平不足 ({boundary.expertise_level:.2f})"

            return True, f"领域允许，专业水平: {boundary.expertise_level:.2f}"
        else:
            # 未知领域，根据配置判断
            if domain in self.profile.ALLOWED_DOMAINS:
                return True, "领域在允许列表中"
            elif domain in self.profile.FORBIDDEN_DOMAINS:
                return False, "领域在禁止列表中"
            else:
                return False, f"领域 '{domain}' 未在配置中定义"

    def update_performance(self, outcome: Dict[str, Any]):
        """
        更新性能记录

        Args:
            outcome: 任务执行结果
        """
        try:
            # 创建性能记录
            record = PerformanceRecord(
                timestamp=datetime.now(),
                task_type=outcome.get("task_type", "general"),
                complexity=outcome.get("complexity", 0.5),
                success=outcome.get("success", False),
                response_time=outcome.get("response_time", 0.0),
                confidence_before=outcome.get("confidence_before", 0.5),
                confidence_after=outcome.get("confidence_after", 0.5),
                user_feedback=outcome.get("user_feedback", None),
                notes=outcome.get("notes", "")
            )

            # 添加到历史记录
            self.performance_history.append(record)

            # 更新统计信息
            self.total_interactions += 1
            if record.success:
                self.total_successes += 1
            else:
                self.total_failures += 1

            # 更新相关能力档案
            capability_domain = self._map_task_to_capability(record.task_type)
            if capability_domain in self.capabilities:
                self._update_capability_profile(capability_domain, record)

            # 更新整体成功率
            overall_success_rate = self._calculate_overall_success_rate()
            self.performance_baseline["success_rate"] = overall_success_rate

            # 定期自我反思
            self.reflection_count += 1
            if self.reflection_count % self.profile.SELF_REFLECTION_INTERVAL == 0:
                self._perform_self_reflection()

        except Exception as e:
            self.logger.error(f"❌ 更新性能记录失败: {str(e)}")

    def _update_capability_profile(self, domain: CapabilityDomain, record: PerformanceRecord):
        """更新能力档案"""
        if domain not in self.capabilities:
            return

        profile = self.capabilities[domain]

        # 更新使用次数
        profile.experience_count += 1

        # 更新时间
        profile.last_used = record.timestamp

        # 更新成功率（指数移动平均）
        alpha = self.adaptation_rate
        old_success_rate = profile.success_rate
        new_success = 1.0 if record.success else 0.0
        profile.success_rate = alpha * new_success + (1 - alpha) * old_success_rate

        # 更新平均表现
        old_avg = profile.avg_performance
        task_performance = 1.0 if record.success else 0.5
        profile.avg_performance = alpha * task_performance + (1 - alpha) * old_avg

        # 更新熟练度（基于长期表现）
        if profile.experience_count >= 5:
            # 只有足够经验后才调整熟练度
            performance_gap = profile.avg_performance - profile.proficiency
            adjustment = performance_gap * alpha * self.learning_momentum
            profile.proficiency = max(0.1, min(1.0, profile.proficiency + adjustment))

        # 更新自信度
        confidence_gap = profile.success_rate - profile.confidence
        profile.confidence = alpha * profile.success_rate + (1 - alpha) * profile.confidence

    def _calculate_overall_success_rate(self) -> float:
        """计算整体成功率"""
        if self.total_interactions == 0:
            return 0.9  # 默认值

        return self.total_successes / self.total_interactions

    def _perform_self_reflection(self):
        """执行自我反思"""
        self.logger.info("🤔 执行自我反思...")

        reflection_result = {
            "timestamp": datetime.now(),
            "total_interactions": self.total_interactions,
            "overall_success_rate": self._calculate_overall_success_rate(),
            "capability_trends": {},
            "identified_weaknesses": [],
            "identified_strengths": [],
            "adjustments_made": []
        }

        # 分析最近表现趋势
        if len(self.performance_history) >= 10:
            recent_records = list(self.performance_history)[-20:]

            # 计算近期成功率
            recent_successes = sum(1 for r in recent_records if r.success)
            recent_success_rate = recent_successes / len(recent_records)

            reflection_result["recent_success_rate"] = recent_success_rate

            # 识别趋势
            trend = self._analyze_performance_trend(recent_records)
            reflection_result["performance_trend"] = trend.value

            # 根据趋势调整
            if trend == PerformanceTrend.DECLINING:
                self.logger.warning(f"⚠️ 检测到性能下降趋势: {recent_success_rate:.2%}")
                self._apply_performance_decline_adjustments()
                reflection_result["adjustments_made"].append("性能下降调整")

            elif trend == PerformanceTrend.IMPROVING:
                self.logger.info(f"✅ 检测到性能提升趋势: {recent_success_rate:.2%}")
                self._apply_performance_improvement_adjustments()
                reflection_result["adjustments_made"].append("性能提升奖励")

            # 识别优势和劣势
            for domain, profile in self.capabilities.items():
                if profile.experience_count >= 5:
                    if profile.success_rate > 0.9:
                        reflection_result["identified_strengths"].append({
                            "domain": domain.value,
                            "success_rate": round(profile.success_rate, 3)
                        })
                    elif profile.success_rate < 0.7:
                        reflection_result["identified_weaknesses"].append({
                            "domain": domain.value,
                            "success_rate": round(profile.success_rate, 3)
                        })

        # 更新自我意识得分
        self.self_awareness_score = self._recalculate_self_awareness()
        reflection_result["self_awareness_score"] = round(self.self_awareness_score, 3)

        # 保存反思历史
        self.reflection_history.append(reflection_result)
        self.last_self_reflection = datetime.now()

        self.logger.info(f"✨ 自我反思完成 - 自我意识得分: {self.self_awareness_score:.3f}")

    def _analyze_performance_trend(self, records: List[PerformanceRecord]) -> PerformanceTrend:
        """分析性能趋势"""
        if len(records) < 10:
            return PerformanceTrend.STABLE

        # 分成前后两组
        mid = len(records) // 2
        first_half = records[:mid]
        second_half = records[mid:]

        # 计算两组的成功率
        first_success_rate = sum(1 for r in first_half if r.success) / len(first_half)
        second_success_rate = sum(1 for r in second_half if r.success) / len(second_half)

        # 计算变化
        change = second_success_rate - first_success_rate

        # 计算波动性
        all_successes = [1 if r.success else 0 for r in records]
        mean_success = sum(all_successes) / len(all_successes)
        variance = sum((s - mean_success) ** 2 for s in all_successes) / len(all_successes)
        volatility = variance ** 0.5

        # 判断趋势
        if volatility > 0.3:
            return PerformanceTrend.VOLATILE
        elif change > 0.1:
            return PerformanceTrend.IMPROVING
        elif change < -0.1:
            return PerformanceTrend.DECLINING
        else:
            return PerformanceTrend.STABLE

    def _apply_performance_decline_adjustments(self):
        """应用性能下降调整"""
        # 降低学习动量
        self.learning_momentum *= 0.95

        # 适度降低所有能力的熟练度
        for domain in self.capabilities:
            self.capabilities[domain].proficiency *= 0.98

        self.logger.info("📉 已应用性能下降调整")

    def _apply_performance_improvement_adjustments(self):
        """应用性能提升调整"""
        # 提高学习动量
        self.learning_momentum = min(1.5, self.learning_momentum * 1.05)

        # 适度提升所有能力的熟练度
        for domain in self.capabilities:
            old_prof = self.capabilities[domain].proficiency
            new_prof = min(1.0, old_prof * 1.01)
            self.capabilities[domain].proficiency = new_prof

        self.logger.info("📈 已应用性能提升奖励")

    def _recalculate_self_awareness(self) -> float:
        """重新计算自我意识得分"""
        # 基于多个因素

        # 1. 数据充足度
        data_adequacy = min(self.total_interactions / 100.0, 1.0)

        # 2. 能力评估稳定性
        if len(self.capabilities) > 0:
            proficiencies = [p.proficiency for p in self.capabilities.values()]
            mean_prof = sum(proficiencies) / len(proficiencies)
            variance = sum((p - mean_prof) ** 2 for p in proficiencies) / len(proficiencies)
            stability = max(0.0, 1.0 - variance * 2)
        else:
            stability = 0.5

        # 3. 反思深度
        reflection_depth = min(len(self.reflection_history) / 20.0, 1.0)

        # 4. 预测准确性（自信度与实际表现的匹配度）
        if len(self.performance_history) >= 10:
            recent = list(self.performance_history)[-20:]
            prediction_errors = [
                abs(r.confidence_before - (1.0 if r.success else 0.0))
                for r in recent
            ]
            avg_error = sum(prediction_errors) / len(prediction_errors)
            prediction_accuracy = max(0.0, 1.0 - avg_error)
        else:
            prediction_accuracy = 0.7

        # 综合得分
        self_awareness = (
            data_adequacy * 0.2 +
            stability * 0.3 +
            reflection_depth * 0.2 +
            prediction_accuracy * 0.3
        )

        return min(self_awareness, 1.0)

    def get_capability_summary(self, domain: Optional[CapabilityDomain] = None) -> Dict[str, Any]:
        """
        获取能力总结

        Args:
            domain: 指定能力领域，None则返回所有

        Returns:
            能力总结字典
        """
        if domain:
            if domain in self.capabilities:
                return self.capabilities[domain].to_dict()
            else:
                return {"error": f"未知能力领域: {domain}"}
        else:
            return {
                d.value: p.to_dict()
                for d, p in self.capabilities.items()
            }

    def get_knowledge_boundary_summary(self) -> Dict[str, Any]:
        """获取知识边界总结"""
        return {
            domain: boundary.to_dict()
            for domain, boundary in self.knowledge_boundaries.items()
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能总结"""
        recent_records = list(self.performance_history)[-10:] if self.performance_history else []

        if recent_records:
            recent_success_rate = sum(1 for r in recent_records if r.success) / len(recent_records)
            avg_response_time = sum(r.response_time for r in recent_records) / len(recent_records)
        else:
            recent_success_rate = 0.0
            avg_response_time = 0.0

        return {
            "total_interactions": self.total_interactions,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "overall_success_rate": round(self._calculate_overall_success_rate(), 3),
            "recent_success_rate": round(recent_success_rate, 3),
            "avg_response_time": round(avg_response_time, 3),
            "performance_baseline": self.performance_baseline,
            "learning_momentum": round(self.learning_momentum, 3),
            "self_awareness_score": round(self.self_awareness_score, 3),
            "last_reflection": self.last_self_reflection.isoformat(),
            "reflection_count": self.reflection_count
        }

    def get_self_summary(self) -> Dict[str, Any]:
        """获取完整的自我总结"""
        return {
            "identity": {
                "name": self.profile.NAME,
                "version": self.profile.VERSION,
                "personality": self.profile.PERSONALITY
            },
            "capabilities": self.get_capability_summary(),
            "knowledge_boundaries": self.get_knowledge_boundary_summary(),
            "performance": self.get_performance_summary(),
            "statistics": {
                "total_interactions": self.total_interactions,
                "success_rate": round(self._calculate_overall_success_rate(), 3),
                "self_awareness": round(self.self_awareness_score, 3)
            }
        }

    def _map_task_to_capability(self, task_type: str) -> CapabilityDomain:
        """将任务类型映射到能力领域"""
        task_mapping = {
            "conversation": CapabilityDomain.LANGUAGE_UNDERSTANDING,
            "query": CapabilityDomain.LOGICAL_REASONING,
            "code_assistance": CapabilityDomain.CODE_GENERATION,
            "analysis": CapabilityDomain.ANALYTICAL_DEPTH,
            "problem_solving": CapabilityDomain.PROBLEM_SOLVING,
            "creation": CapabilityDomain.CREATIVE_THINKING,
            "data_analysis": CapabilityDomain.DATA_ANALYSIS,
            "task_execution": CapabilityDomain.TOOL_USAGE,
            "learning": CapabilityDomain.ADAPTABILITY,
            "document_processing": CapabilityDomain.LANGUAGE_UNDERSTANDING,
            "clarification": CapabilityDomain.EMOTIONAL_INTELLIGENCE
        }

        return task_mapping.get(task_type, CapabilityDomain.LOGICAL_REASONING)

    def update_knowledge_boundary(self, domain: str, expertise_level: float,
                                  is_allowed: bool = True, notes: str = ""):
        """
        更新知识边界

        Args:
            domain: 领域名称
            expertise_level: 专业水平 (0.0-1.0)
            is_allowed: 是否允许
            notes: 备注
        """
        if domain in self.knowledge_boundaries:
            boundary = self.knowledge_boundaries[domain]
            boundary.expertise_level = expertise_level
            boundary.is_allowed = is_allowed
            boundary.notes = notes
            boundary.last_updated = datetime.now()
        else:
            self.knowledge_boundaries[domain] = KnowledgeBoundary(
                domain=domain,
                expertise_level=expertise_level,
                is_allowed=is_allowed,
                is_forbidden=False,
                notes=notes
            )

        self.logger.info(f"📚 更新知识边界: {domain} (专业水平: {expertise_level:.2f})")

    def reset_statistics(self):
        """重置统计信息（保留能力模型）"""
        self.performance_history.clear()
        self.reflection_history.clear()
        self.total_interactions = 0
        self.total_successes = 0
        self.total_failures = 0
        self.reflection_count = 0
        self.last_self_reflection = datetime.now()

        self.logger.info("🔄 统计信息已重置")

    def export_self_model(self) -> Dict[str, Any]:
        """导出自我模型（用于持久化）"""
        return {
            "capabilities": {
                d.value: p.to_dict()
                for d, p in self.capabilities.items()
            },
            "knowledge_boundaries": {
                domain: b.to_dict()
                for domain, b in self.knowledge_boundaries.items()
            },
            "performance_baseline": self.performance_baseline,
            "statistics": {
                "total_interactions": self.total_interactions,
                "total_successes": self.total_successes,
                "total_failures": self.total_failures,
                "self_awareness_score": self.self_awareness_score,
                "learning_momentum": self.learning_momentum
            },
            "metadata": {
                "last_reflection": self.last_self_reflection.isoformat(),
                "reflection_count": self.reflection_count,
                "export_timestamp": datetime.now().isoformat()
            }
        }

    def import_self_model(self, model_data: Dict[str, Any]):
        """
        导入自我模型（从持久化恢复）

        Args:
            model_data: 导出的自我模型数据
        """
        try:
            # 恢复能力模型
            for domain_str, cap_data in model_data.get("capabilities", {}).items():
                domain = CapabilityDomain(domain_str)
                if domain in self.capabilities:
                    profile = self.capabilities[domain]
                    profile.proficiency = cap_data.get("proficiency", profile.proficiency)
                    profile.confidence = cap_data.get("confidence", profile.confidence)
                    profile.experience_count = cap_data.get("experience_count", 0)
                    profile.success_rate = cap_data.get("success_rate", 0.9)
                    profile.avg_performance = cap_data.get("avg_performance", 0.85)

            # 恢复知识边界
            for domain, bound_data in model_data.get("knowledge_boundaries", {}).items():
                if domain in self.knowledge_boundaries:
                    boundary = self.knowledge_boundaries[domain]
                    boundary.expertise_level = bound_data.get("expertise_level", 0.8)
                    boundary.is_allowed = bound_data.get("is_allowed", True)
                    boundary.notes = bound_data.get("notes", "")

            # 恢复统计信息
            stats = model_data.get("statistics", {})
            self.total_interactions = stats.get("total_interactions", 0)
            self.total_successes = stats.get("total_successes", 0)
            self.total_failures = stats.get("total_failures", 0)
            self.self_awareness_score = stats.get("self_awareness_score", 0.7)
            self.learning_momentum = stats.get("learning_momentum", 1.0)

            self.logger.info("✅ 自我模型已导入")

        except Exception as e:
            self.logger.error(f"❌ 导入自我模型失败: {str(e)}")
