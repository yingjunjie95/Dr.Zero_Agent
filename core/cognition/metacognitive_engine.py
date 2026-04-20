"""
元认知引擎 - Metacognitive Engine
负责Agent的自我监控、评估和调节能力
实现真正的自主认知，包括自我意识、风险评估和策略优化
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import deque


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConfidenceLevel(Enum):
    """置信度等级"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class CognitiveState(Enum):
    """认知状态"""
    CLEAR = "clear"  # 清晰
    UNCERTAIN = "uncertain"  # 不确定
    CONFUSED = "confused"  # 困惑
    OVERLOADED = "overloaded"  # 过载
    FOCUSED = "focused"  # 专注


@dataclass
class MetacognitiveAssessment:
    """元认知评估结果"""
    overall_confidence: float  # 整体置信度 (0.0-1.0)
    confidence_level: ConfidenceLevel
    risk_level: RiskLevel
    cognitive_state: CognitiveState

    # 自我评估
    self_awareness_score: float  # 自我意识得分
    knowledge_gaps: List[str]  # 知识缺口
    capability_limits: List[str]  # 能力边界

    # 情境评估
    task_complexity: float  # 任务复杂度 (0.0-1.0)
    resource_adequacy: float  # 资源充足度 (0.0-1.0)
    time_pressure: float  # 时间压力 (0.0-1.0)

    # 决策支持
    has_goal: bool  # 是否有明确目标
    goal_description: str  # 目标描述
    requires_clarification: bool  # 是否需要澄清
    clarification_points: List[str]  # 需要澄清的点

    # 风险控制
    potential_risks: List[Dict[str, Any]]  # 潜在风险列表
    mitigation_strategies: List[str]  # 缓解策略

    # 情绪影响
    emotional_impact: float  # 情绪对决策的影响程度 (0.0-1.0)
    empathy_needed: bool  # 是否需要共情

    # 元数据
    assessment_timestamp: datetime = field(default_factory=datetime.now)
    reasoning_trace: List[str] = field(default_factory=list)  # 推理轨迹

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "overall_confidence": round(self.overall_confidence, 3),
            "confidence_level": self.confidence_level.value,
            "risk_level": self.risk_level.value,
            "cognitive_state": self.cognitive_state.value,
            "self_awareness_score": round(self.self_awareness_score, 3),
            "knowledge_gaps": self.knowledge_gaps,
            "capability_limits": self.capability_limits,
            "task_complexity": round(self.task_complexity, 3),
            "resource_adequacy": round(self.resource_adequacy, 3),
            "time_pressure": round(self.time_pressure, 3),
            "has_goal": self.has_goal,
            "goal_description": self.goal_description,
            "requires_clarification": self.requires_clarification,
            "clarification_points": self.clarification_points,
            "potential_risks": self.potential_risks,
            "mitigation_strategies": self.mitigation_strategies,
            "emotional_impact": round(self.emotional_impact, 3),
            "empathy_needed": self.empathy_needed,
            "reasoning_trace": self.reasoning_trace
        }


class SelfModel:
    """
    自我模型 - Agent的自我认知表示
    维护Agent对自身能力、知识和局限性的理解
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
        self.capabilities = self._build_capability_model()

        # 知识边界
        self.knowledge_domains = agent_profile.ALLOWED_DOMAINS.copy()
        self.forbidden_domains = agent_profile.FORBIDDEN_DOMAINS.copy()

        # 性能基线
        self.performance_baseline = {
            "avg_response_time": 2.0,
            "success_rate": 0.95,
            "user_satisfaction": 0.85
        }

        # 历史表现
        self.recent_performance: deque = deque(maxlen=50)

        # 自我认知更新
        self.last_self_reflection = datetime.now()
        self.reflection_count = 0

        self.logger.info("✅ 自我模型初始化完成")

    def _build_capability_model(self) -> Dict[str, float]:
        """构建能力模型"""
        return {
            "language_understanding": 0.9,
            "logical_reasoning": 0.85,
            "creative_thinking": self.profile.CREATIVITY_LEVEL,
            "analytical_depth": self.profile.ANALYTICAL_DEPTH,
            "emotional_intelligence": self.profile.EMPATHY_LEVEL,
            "memory_recall": 0.88,
            "tool_usage": 0.82,
            "adaptability": 0.75
        }

    def assess_capability(self, task_type: str, complexity: float) -> float:
        """
        评估处理特定任务的能力

        Args:
            task_type: 任务类型
            complexity: 任务复杂度 (0.0-1.0)

        Returns:
            能力得分 (0.0-1.0)
        """
        # 基础能力得分
        base_capability = self.capabilities.get(task_type, 0.7)

        # 复杂度衰减
        complexity_factor = max(0.3, 1.0 - complexity * 0.5)

        # 考虑最近表现
        recent_success_rate = self._get_recent_success_rate()

        # 综合得分
        capability_score = base_capability * complexity_factor * recent_success_rate

        return min(capability_score, 1.0)

    def check_domain_permission(self, domain: str) -> Tuple[bool, str]:
        """
        检查领域权限

        Args:
            domain: 任务领域

        Returns:
            (是否允许, 原因说明)
        """
        if domain in self.forbidden_domains:
            return False, f"领域 '{domain}' 在禁止列表中"

        if domain not in self.knowledge_domains:
            return False, f"领域 '{domain}' 超出知识范围"

        return True, "领域允许"

    def update_performance(self, outcome: Dict[str, Any]):
        """
        更新性能记录

        Args:
            outcome: 任务执行结果
        """
        performance_record = {
            "timestamp": datetime.now(),
            "success": outcome.get("success", False),
            "response_time": outcome.get("response_time", 0.0),
            "confidence": outcome.get("confidence", 0.5),
            "user_feedback": outcome.get("user_feedback", None)
        }

        self.recent_performance.append(performance_record)

        # 定期自我反思
        self.reflection_count += 1
        if self.reflection_count % self.profile.SELF_REFLECTION_INTERVAL == 0:
            self._perform_self_reflection()

    def _get_recent_success_rate(self) -> float:
        """获取最近的成功率"""
        if not self.recent_performance:
            return 0.9  # 默认值

        successes = sum(1 for r in self.recent_performance if r["success"])
        return successes / len(self.recent_performance)

    def _perform_self_reflection(self):
        """执行自我反思"""
        self.logger.info("🤔 执行自我反思...")

        # 分析最近表现趋势
        if len(self.recent_performance) >= 10:
            recent_records = list(self.recent_performance)[-10:]
            success_rate = sum(1 for r in recent_records if r["success"]) / len(recent_records)

            # 更新能力模型
            if success_rate < 0.7:
                self.logger.warning(f"⚠️ 最近成功率较低: {success_rate:.2%}")
                # 降低相关能力评分
                for capability in self.capabilities:
                    self.capabilities[capability] *= 0.95

            elif success_rate > 0.9:
                self.logger.info(f"✅ 最近表现优秀: {success_rate:.2%}")
                # 适度提升能力评分
                for capability in self.capabilities:
                    self.capabilities[capability] = min(
                        self.capabilities[capability] * 1.02, 1.0
                    )

        self.last_self_reflection = datetime.now()
        self.logger.info("✨ 自我反思完成")

    def get_self_summary(self) -> Dict[str, Any]:
        """获取自我总结"""
        return {
            "capabilities": self.capabilities.copy(),
            "knowledge_domains": self.knowledge_domains,
            "forbidden_domains": self.forbidden_domains,
            "recent_success_rate": self._get_recent_success_rate(),
            "performance_baseline": self.performance_baseline,
            "last_reflection": self.last_self_reflection.isoformat()
        }


class MetacognitiveEngine:
    """
    元认知引擎

    功能：
    - 情境评估：分析当前任务的复杂度和风险
    - 自我监控：评估自身能力和置信度
    - 风险识别：发现潜在问题和边界情况
    - 策略建议：为决策提供元认知支持
    - 学习优化：从经验中改进元认知能力
    """

    def __init__(self, self_model: SelfModel, risk_threshold: float = 0.7):
        """
        初始化元认知引擎

        Args:
            self_model: Agent自我模型
            risk_threshold: 风险阈值
        """
        self.self_model = self_model
        self.risk_threshold = risk_threshold
        self.logger = logging.getLogger("MetacognitiveEngine")

        # 元认知历史
        self.assessment_history: deque = deque(maxlen=100)

        # 模式识别
        self.recognized_patterns: Dict[str, int] = {}

        # 校准参数
        self.confidence_calibration_factor = 1.0

        self.logger.info(f"✅ 元认知引擎初始化完成 - 风险阈值: {risk_threshold}")

    def evaluate_situation(
        self,
        user_input: str,
        context: Dict[str, Any],
        available_resources: Dict[str, Any],
        relevant_memories: List[Any],
        user_intent: Any = None,
        user_emotion: Any = None
    ) -> Dict[str, Any]:
        """
        全面评估当前情境

        Args:
            user_input: 用户输入
            context: 对话上下文
            available_resources: 可用资源
            relevant_memories: 相关记忆
            user_intent: 用户意图分析结果
            user_emotion: 用户情绪检测结果

        Returns:
            元认知评估结果字典
        """
        try:
            self.logger.debug("🧠 开始元认知评估...")

            reasoning_trace = []

            # 1. 评估任务复杂度
            task_complexity = self._assess_task_complexity(
                user_input, user_intent, context
            )
            reasoning_trace.append(f"任务复杂度评估: {task_complexity:.2f}")

            # 2. 评估资源充足度
            resource_adequacy = self._assess_resource_adequacy(
                available_resources, task_complexity
            )
            reasoning_trace.append(f"资源充足度: {resource_adequacy:.2f}")

            # 3. 评估时间压力
            time_pressure = self._assess_time_pressure(
                user_intent, available_resources
            )
            reasoning_trace.append(f"时间压力: {time_pressure:.2f}")

            # 4. 自我能力评估
            capability_score = self._assess_self_capability(
                user_input, user_intent, task_complexity
            )
            reasoning_trace.append(f"能力得分: {capability_score:.2f}")

            # 5. 计算整体置信度
            overall_confidence = self._calculate_overall_confidence(
                capability_score, task_complexity, resource_adequacy,
                len(relevant_memories)
            )
            reasoning_trace.append(f"整体置信度: {overall_confidence:.2f}")

            # 6. 风险评估
            risk_level, potential_risks = self._assess_risks(
                user_input, user_intent, user_emotion,
                task_complexity, overall_confidence
            )
            reasoning_trace.append(f"风险等级: {risk_level.value}")

            # 7. 识别知识缺口
            knowledge_gaps = self._identify_knowledge_gaps(
                user_input, user_intent, relevant_memories
            )

            # 8. 检查能力边界
            capability_limits = self._check_capability_limits(
                user_input, user_intent
            )

            # 9. 确定认知状态
            cognitive_state = self._determine_cognitive_state(
                overall_confidence, task_complexity, resource_adequacy
            )

            # 10. 评估情绪影响
            emotional_impact, empathy_needed = self._assess_emotional_impact(
                user_emotion
            )

            # 11. 目标识别
            has_goal, goal_description = self._identify_goal(
                user_input, user_intent
            )

            # 12. 澄清需求判断
            requires_clarification, clarification_points = self._check_clarification_needed(
                user_input, user_intent, overall_confidence, knowledge_gaps
            )

            # 13. 生成缓解策略
            mitigation_strategies = self._generate_mitigation_strategies(
                potential_risks, knowledge_gaps, capability_limits
            )

            # 创建评估结果
            confidence_level = self._map_confidence_level(overall_confidence)

            assessment = MetacognitiveAssessment(
                overall_confidence=overall_confidence,
                confidence_level=confidence_level,
                risk_level=risk_level,
                cognitive_state=cognitive_state,
                self_awareness_score=self._calculate_self_awareness(),
                knowledge_gaps=knowledge_gaps,
                capability_limits=capability_limits,
                task_complexity=task_complexity,
                resource_adequacy=resource_adequacy,
                time_pressure=time_pressure,
                has_goal=has_goal,
                goal_description=goal_description,
                requires_clarification=requires_clarification,
                clarification_points=clarification_points,
                potential_risks=potential_risks,
                mitigation_strategies=mitigation_strategies,
                emotional_impact=emotional_impact,
                empathy_needed=empathy_needed,
                reasoning_trace=reasoning_trace
            )

            # 记录历史
            self.assessment_history.append(assessment)

            # 模式识别
            self._recognize_patterns(assessment)

            self.logger.debug("✨ 元认知评估完成")
            return assessment.to_dict()

        except Exception as e:
            self.logger.error(f"❌ 元认知评估失败: {str(e)}", exc_info=True)
            return self._create_fallback_assessment()

    def _assess_task_complexity(
        self,
        user_input: str,
        user_intent: Any,
        context: Dict[str, Any]
    ) -> float:
        """评估任务复杂度"""
        complexity = 0.0

        # 基于输入长度
        word_count = len(user_input.split())
        if word_count > 100:
            complexity += 0.3
        elif word_count > 50:
            complexity += 0.2
        elif word_count > 20:
            complexity += 0.1

        # 基于意图类型
        if user_intent:
            intent_complexity_map = {
                "conversation": 0.1,
                "query": 0.3,
                "clarification": 0.2,
                "code_assistance": 0.6,
                "analysis": 0.6,
                "problem_solving": 0.7,
                "creation": 0.5,
                "data_analysis": 0.7,
                "task_execution": 0.5,
                "learning": 0.4,
                "document_processing": 0.5,
                "unknown": 0.5
            }

            intent_type = getattr(user_intent, 'intent_type', None)
            if intent_type:
                complexity += intent_complexity_map.get(intent_type.value, 0.4)

        # 基于上下文长度
        if context and len(context) > 10:
            complexity += 0.2

        # 基于特殊要求
        complex_indicators = ["详细", "全面", "深入", "复杂", "多步骤"]
        if any(indicator in user_input for indicator in complex_indicators):
            complexity += 0.15

        return min(complexity, 1.0)

    def _assess_resource_adequacy(
        self,
        available_resources: Dict[str, Any],
        task_complexity: float
    ) -> float:
        """评估资源充足度"""
        if not available_resources:
            return 0.5  # 默认中等

        # 内存充足度
        memory_usage = available_resources.get('memory_usage', 0.5)
        memory_adequacy = max(0.0, 1.0 - memory_usage)

        # CPU充足度
        cpu_load = available_resources.get('cpu_load', 0.5)
        cpu_adequacy = max(0.0, 1.0 - cpu_load)

        # 综合充足度（考虑任务复杂度）
        base_adequacy = (memory_adequacy * 0.6 + cpu_adequacy * 0.4)

        # 高复杂度任务对资源要求更高
        if task_complexity > 0.7:
            adequacy = base_adequacy * 0.8
        else:
            adequacy = base_adequacy

        return min(adequacy, 1.0)

    def _assess_time_pressure(
        self,
        user_intent: Any,
        available_resources: Dict[str, Any]
    ) -> float:
        """评估时间压力"""
        pressure = 0.0

        # 基于紧急程度
        if user_intent:
            urgency = getattr(user_intent, 'urgency', None)
            if urgency:
                urgency_map = {
                    "low": 0.1,
                    "normal": 0.3,
                    "high": 0.7,
                    "critical": 0.9
                }
                pressure += urgency_map.get(urgency.value, 0.3)

        # 基于系统负载
        if available_resources:
            cpu_load = available_resources.get('cpu_load', 0.5)
            if cpu_load > 0.8:
                pressure += 0.2

        return min(pressure, 1.0)

    def _assess_self_capability(
        self,
        user_input: str,
        user_intent: Any,
        task_complexity: float
    ) -> float:
        """评估自身处理能力"""
        # 确定任务类型
        task_type = "general"
        if user_intent:
            intent_type = getattr(user_intent, 'intent_type', None)
            if intent_type:
                task_type = intent_type.value

        # 使用自我模型评估
        capability = self.self_model.assess_capability(task_type, task_complexity)

        # 检查领域权限
        if user_intent:
            domain = getattr(user_intent, 'domain', 'general')
            allowed, reason = self.self_model.check_domain_permission(domain)
            if not allowed:
                self.logger.warning(f"⚠️ 领域限制: {reason}")
                capability *= 0.3  # 大幅降低能力得分

        return capability

    def _calculate_overall_confidence(
        self,
        capability_score: float,
        task_complexity: float,
        resource_adequacy: float,
        memory_count: int
    ) -> float:
        """计算整体置信度"""
        # 基础置信度
        confidence = capability_score * 0.4

        # 复杂度影响（负相关）
        complexity_factor = max(0.3, 1.0 - task_complexity * 0.5)
        confidence *= complexity_factor

        # 资源影响
        resource_factor = 0.7 + resource_adequacy * 0.3
        confidence *= resource_factor

        # 记忆支持
        memory_bonus = min(memory_count * 0.02, 0.15)
        confidence += memory_bonus

        # 应用校准因子
        confidence *= self.confidence_calibration_factor

        return min(max(confidence, 0.0), 1.0)

    def _assess_risks(
        self,
        user_input: str,
        user_intent: Any,
        user_emotion: Any,
        task_complexity: float,
        confidence: float
    ) -> Tuple[RiskLevel, List[Dict[str, Any]]]:
        """评估风险"""
        risks = []
        risk_score = 0.0

        # 1. 置信度风险
        if confidence < 0.4:
            risks.append({
                "type": "low_confidence",
                "severity": "high",
                "description": f"置信度过低: {confidence:.2f}"
            })
            risk_score += 0.3

        # 2. 复杂度风险
        if task_complexity > 0.7:
            risks.append({
                "type": "high_complexity",
                "severity": "medium",
                "description": f"任务复杂度高: {task_complexity:.2f}"
            })
            risk_score += 0.2

        # 3. 领域风险
        if user_intent:
            domain = getattr(user_intent, 'domain', 'general')
            if domain in self.self_model.forbidden_domains:
                risks.append({
                    "type": "forbidden_domain",
                    "severity": "critical",
                    "description": f"涉及禁止领域: {domain}"
                })
                risk_score += 0.5

        # 4. 情绪风险
        if user_emotion:
            emotion_type = getattr(user_emotion, 'primary_emotion', None)
            intensity = getattr(user_emotion, 'emotion_intensity', 0.0)

            if emotion_type and emotion_type.value in ['anger', 'frustration']:
                if intensity > 0.6:
                    risks.append({
                        "type": "negative_emotion",
                        "severity": "medium",
                        "description": f"用户负面情绪强烈: {emotion_type.value}"
                    })
                    risk_score += 0.15

        # 5. 输入异常风险
        if len(user_input) > 1000:
            risks.append({
                "type": "long_input",
                "severity": "low",
                "description": "输入过长，可能包含复杂需求"
            })
            risk_score += 0.1

        # 确定风险等级
        if risk_score >= 0.6:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 0.4:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 0.2:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        return risk_level, risks

    def _identify_knowledge_gaps(
        self,
        user_input: str,
        user_intent: Any,
        relevant_memories: List[Any]
    ) -> List[str]:
        """识别知识缺口"""
        gaps = []

        # 检查是否有相关记忆
        if not relevant_memories or len(relevant_memories) == 0:
            gaps.append("缺乏相关历史经验")

        # 检查特定领域关键词
        specialized_domains = {
            "量子计算": ["量子", "quantum"],
            "区块链": ["区块链", "blockchain", "加密货币"],
            "机器学习": ["神经网络", "深度学习", "训练模型"]
        }

        for domain, keywords in specialized_domains.items():
            if any(kw in user_input for kw in keywords):
                if domain not in self.self_model.knowledge_domains:
                    gaps.append(f"专业领域知识可能不足: {domain}")

        return gaps

    def _check_capability_limits(
        self,
        user_input: str,
        user_intent: Any
    ) -> List[str]:
        """检查能力边界"""
        limits = []

        # 检查是否需要实时信息
        real_time_indicators = ["最新", "今天", "现在", "当前", "real-time"]
        if any(indicator in user_input for indicator in real_time_indicators):
            limits.append("无法访问实时信息")

        # 检查是否需要外部工具
        if user_intent:
            suggested_tools = getattr(user_intent, 'suggested_tools', [])
            if suggested_tools:
                limits.append(f"需要调用外部工具: {', '.join(suggested_tools[:3])}")

        # 检查复杂度是否超出限制
        if user_intent:
            complexity = getattr(user_intent, 'complexity', None)
            if complexity and complexity.value >= 4:  # VERY_COMPLEX
                limits.append("任务复杂度接近处理上限")

        return limits

    def _determine_cognitive_state(
        self,
        confidence: float,
        complexity: float,
        resources: float
    ) -> CognitiveState:
        """确定认知状态"""
        if confidence > 0.7 and complexity < 0.5:
            return CognitiveState.CLEAR
        elif confidence > 0.6 and resources > 0.6:
            return CognitiveState.FOCUSED
        elif confidence < 0.3:
            return CognitiveState.CONFUSED
        elif resources < 0.3:
            return CognitiveState.OVERLOADED
        else:
            return CognitiveState.UNCERTAIN

    def _assess_emotional_impact(
        self,
        user_emotion: Any
    ) -> Tuple[float, bool]:
        """评估情绪影响"""
        if not user_emotion:
            return 0.0, False

        intensity = getattr(user_emotion, 'emotion_intensity', 0.0)
        primary_emotion = getattr(user_emotion, 'primary_emotion', None)
        requires_empathy = getattr(user_emotion, 'requires_empathy', False)

        # 高强度情绪会影响决策
        if intensity > 0.7:
            impact = 0.6
        elif intensity > 0.4:
            impact = 0.3
        else:
            impact = 0.1

        # 负面情绪影响更大
        if primary_emotion and primary_emotion.value in [
            'anger', 'sadness', 'fear', 'frustration'
        ]:
            impact *= 1.3

        return min(impact, 1.0), requires_empathy

    def _identify_goal(
        self,
        user_input: str,
        user_intent: Any
    ) -> Tuple[bool, str]:
        """识别用户目标"""
        if not user_intent:
            return False, ""

        # 从意图中提取目标
        intent_type = getattr(user_intent, 'intent_type', None)
        if intent_type and intent_type.value != 'conversation':
            return True, f"完成{intent_type.value}任务"

        # 从输入中检测目标关键词
        goal_indicators = ["帮我", "我需要", "想要", "目的是", "目标是"]
        for indicator in goal_indicators:
            if indicator in user_input:
                return True, f"响应用户请求"

        return False, ""

    def _check_clarification_needed(
        self,
        user_input: str,
        user_intent: Any,
        confidence: float,
        knowledge_gaps: List[str]
    ) -> Tuple[bool, List[str]]:
        """判断是否需要澄清"""
        clarification_points = []

        # 置信度过低
        if confidence < self.self_model.profile.SELF_DOUBT_THRESHOLD:
            clarification_points.append("置信度不足，需要更多信息")

        # 意图分析表明需要澄清
        if user_intent:
            needs_clarification = getattr(user_intent, 'requires_clarification', False)
            if needs_clarification:
                questions = getattr(user_intent, 'clarification_questions', [])
                clarification_points.extend(questions)

        # 存在知识缺口
        if knowledge_gaps:
            clarification_points.append(f"知识缺口: {'; '.join(knowledge_gaps[:2])}")

        requires = len(clarification_points) > 0
        return requires, clarification_points

    def _generate_mitigation_strategies(
        self,
        risks: List[Dict[str, Any]],
        knowledge_gaps: List[str],
        capability_limits: List[str]
    ) -> List[str]:
        """生成缓解策略"""
        strategies = []

        # 针对风险
        for risk in risks:
            if risk['type'] == 'low_confidence':
                strategies.append("采用保守策略，提供备选方案")
            elif risk['type'] == 'forbidden_domain':
                strategies.append("礼貌拒绝并提供替代方向")
            elif risk['type'] == 'high_complexity':
                strategies.append("分解任务，分步骤处理")

        # 针对知识缺口
        if knowledge_gaps:
            strategies.append("明确说明知识局限性")
            strategies.append("建议用户查阅专业资料")

        # 针对能力限制
        if capability_limits:
            strategies.append("如实告知能力边界")
            strategies.append("寻求用户协作或简化需求")

        # 通用策略
        if not strategies:
            strategies.append("按标准流程处理")

        return strategies

    def _calculate_self_awareness(self) -> float:
        """计算自我意识得分"""
        # 基于评估历史的稳定性
        if len(self.assessment_history) < 5:
            return 0.7  # 默认值

        recent_assessments = list(self.assessment_history)[-10:]
        confidences = [a.overall_confidence for a in recent_assessments]

        # 计算置信度的标准差（越低越稳定）
        mean_confidence = sum(confidences) / len(confidences)
        variance = sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences)
        std_dev = variance ** 0.5

        # 稳定性得分（标准差越小，稳定性越高）
        stability_score = max(0.0, 1.0 - std_dev * 2)

        # 结合历史记录数量
        history_factor = min(len(self.assessment_history) / 50.0, 1.0)

        # 综合自我意识得分
        self_awareness = stability_score * 0.6 + history_factor * 0.4

        return min(self_awareness, 1.0)

    def _recognize_patterns(self, assessment: MetacognitiveAssessment):
        """识别元认知模式"""
        # 基于认知状态的模式
        state_key = f"state_{assessment.cognitive_state.value}"
        self.recognized_patterns[state_key] = self.recognized_patterns.get(state_key, 0) + 1

        # 基于风险等级的模式
        risk_key = f"risk_{assessment.risk_level.value}"
        self.recognized_patterns[risk_key] = self.recognized_patterns.get(risk_key, 0) + 1

        # 基于置信度区间的模式
        if assessment.overall_confidence > 0.8:
            conf_key = "confidence_high"
        elif assessment.overall_confidence > 0.5:
            conf_key = "confidence_medium"
        else:
            conf_key = "confidence_low"
        self.recognized_patterns[conf_key] = self.recognized_patterns.get(conf_key, 0) + 1

    def _create_fallback_assessment(self) -> Dict[str, Any]:
        """创建降级评估结果（异常情况下使用）"""
        self.logger.warning("⚠️ 使用降级评估结果")

        fallback = MetacognitiveAssessment(
            overall_confidence=0.5,
            confidence_level=ConfidenceLevel.MODERATE,
            risk_level=RiskLevel.MEDIUM,
            cognitive_state=CognitiveState.UNCERTAIN,
            self_awareness_score=0.6,
            knowledge_gaps=["评估过程出现异常"],
            capability_limits=["元认知引擎功能受限"],
            task_complexity=0.5,
            resource_adequacy=0.5,
            time_pressure=0.3,
            has_goal=False,
            goal_description="",
            requires_clarification=True,
            clarification_points=["系统评估异常，请重新描述需求"],
            potential_risks=[{
                "type": "system_error",
                "severity": "medium",
                "description": "元认知评估失败"
            }],
            mitigation_strategies=["重试评估", "采用保守策略"],
            emotional_impact=0.0,
            empathy_needed=False,
            reasoning_trace=["评估过程异常，启用降级方案"]
        )

        return fallback.to_dict()

    def get_metacognitive_summary(self) -> Dict[str, Any]:
        """获取元认知总结"""
        if not self.assessment_history:
            return {
                "total_assessments": 0,
                "average_confidence": 0.0,
                "risk_distribution": {},
                "state_distribution": {},
                "pattern_recognition": {}
            }

        assessments = list(self.assessment_history)

        # 平均置信度
        avg_confidence = sum(a.overall_confidence for a in assessments) / len(assessments)

        # 风险分布
        risk_dist = {}
        for a in assessments:
            risk_val = a.risk_level.value
            risk_dist[risk_val] = risk_dist.get(risk_val, 0) + 1

        # 认知状态分布
        state_dist = {}
        for a in assessments:
            state_val = a.cognitive_state.value
            state_dist[state_val] = state_dist.get(state_val, 0) + 1

        return {
            "total_assessments": len(assessments),
            "average_confidence": round(avg_confidence, 3),
            "risk_distribution": risk_dist,
            "state_distribution": state_dist,
            "pattern_recognition": self.recognized_patterns.copy(),
            "calibration_factor": self.confidence_calibration_factor
        }

    def calibrate_confidence(self, actual_outcomes: List[Dict[str, Any]]):
        """
        校准置信度

        Args:
            actual_outcomes: 实际结果列表，包含 predicted_confidence 和 actual_success
        """
        if not actual_outcomes or len(actual_outcomes) < 10:
            return

        # 计算预测置信度与实际成功率的相关性
        predicted_confs = [o.get('predicted_confidence', 0.5) for o in actual_outcomes]
        actual_successes = [1 if o.get('actual_success', False) else 0 for o in actual_outcomes]

        # 简单线性回归计算校准因子
        n = len(actual_outcomes)
        sum_pred = sum(predicted_confs)
        sum_actual = sum(actual_successes)
        sum_prod = sum(p * a for p, a in zip(predicted_confs, actual_successes))
        sum_pred_sq = sum(p ** 2 for p in predicted_confs)

        # 计算斜率（校准因子）
        denominator = n * sum_pred_sq - sum_pred ** 2
        if denominator != 0:
            slope = (n * sum_prod - sum_pred * sum_actual) / denominator

            # 更新校准因子（平滑更新）
            old_factor = self.confidence_calibration_factor
            new_factor = slope
            self.confidence_calibration_factor = old_factor * 0.8 + new_factor * 0.2

            # 限制在合理范围内
            self.confidence_calibration_factor = max(0.5, min(1.5, self.confidence_calibration_factor))

            self.logger.info(
                f"📊 置信度校准: {old_factor:.3f} → {self.confidence_calibration_factor:.3f}"
            )

    def adapt_strategy(self, assessment: MetacognitiveAssessment) -> Dict[str, Any]:
        """
        根据元认知评估调整策略

        Args:
            assessment: 元认知评估结果

        Returns:
            策略建议
        """
        strategy = {
            "response_mode": "standard",
            "detail_level": "normal",
            "needs_verification": False,
            "should_ask_clarification": False,
            "requires_human_oversight": False,
            "suggested_approach": []
        }

        # 基于风险等级
        if assessment.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            strategy["response_mode"] = "conservative"
            strategy["needs_verification"] = True
            strategy["suggested_approach"].append("采用保守策略，验证关键信息")

            if assessment.risk_level == RiskLevel.CRITICAL:
                strategy["requires_human_oversight"] = True
                strategy["suggested_approach"].append("建议人工审核")

        # 基于置信度
        if assessment.overall_confidence < 0.4:
            strategy["response_mode"] = "cautious"
            strategy["should_ask_clarification"] = True
            strategy["suggested_approach"].append("请求用户澄清")
            strategy["detail_level"] = "brief"
        elif assessment.overall_confidence > 0.8:
            strategy["response_mode"] = "confident"
            strategy["detail_level"] = "comprehensive"
            strategy["suggested_approach"].append("提供详细解答")

        # 基于认知状态
        if assessment.cognitive_state == CognitiveState.CONFUSED:
            strategy["should_ask_clarification"] = True
            strategy["suggested_approach"].append("表达困惑并寻求澄清")
        elif assessment.cognitive_state == CognitiveState.OVERLOADED:
            strategy["detail_level"] = "concise"
            strategy["suggested_approach"].append("简化响应，分步处理")

        # 基于知识缺口
        if assessment.knowledge_gaps:
            strategy["needs_verification"] = True
            strategy["suggested_approach"].append("明确说明知识局限")

        # 基于情绪影响
        if assessment.empathy_needed:
            strategy["suggested_approach"].append("优先共情回应")
            strategy["response_mode"] = "empathetic"

        # 基于时间压力
        if assessment.time_pressure > 0.7:
            strategy["detail_level"] = "concise"
            strategy["suggested_approach"].append("快速响应，抓住重点")

        return strategy

    def learn_from_feedback(self, feedback: Dict[str, Any]):
        """
        从反馈中学习

        Args:
            feedback: 用户或系统反馈
        """
        try:
            # 提取反馈信息
            success = feedback.get('success', None)
            satisfaction = feedback.get('satisfaction', None)
            correction = feedback.get('correction', None)

            # 更新性能记录
            if success is not None:
                performance_record = {
                    "timestamp": datetime.now(),
                    "success": success,
                    "satisfaction": satisfaction,
                    "has_correction": correction is not None
                }

                # 通知自我模型更新
                self.self_model.update_performance(performance_record)

                # 如果有纠正信息，调整相关参数
                if correction:
                    self._apply_correction(correction)

                    self.logger.info("✅ 已从反馈中学习")

        except Exception as e:
            self.logger.error(f"❌ 从反馈学习失败: {str(e)}")

    def _apply_correction(self, correction: Dict[str, Any]):
        """应用纠正"""
        correction_type = correction.get('type', '')

        if correction_type == 'confidence_adjustment':
            # 置信度调整
            direction = correction.get('direction', 'none')
            if direction == 'overconfident':
                self.confidence_calibration_factor *= 0.95
                self.logger.info("📉 检测到过度自信，降低校准因子")
            elif direction == 'underconfident':
                self.confidence_calibration_factor *= 1.05
                self.logger.info("📈 检测到自信不足，提高校准因子")

        elif correction_type == 'capability_update':
            # 能力模型更新
            capability = correction.get('capability', '')
            adjustment = correction.get('adjustment', 0.0)

            if capability in self.self_model.capabilities:
                old_value = self.self_model.capabilities[capability]
                new_value = max(0.0, min(1.0, old_value + adjustment))
                self.self_model.capabilities[capability] = new_value

                self.logger.info(
                    f"🔄 能力更新: {capability} {old_value:.2f} → {new_value:.2f}"
                )

    def reset(self):
        """重置元认知引擎状态"""
        self.assessment_history.clear()
        self.recognized_patterns.clear()
        self.confidence_calibration_factor = 1.0
        self.logger.info("🔄 元认知引擎已重置")
