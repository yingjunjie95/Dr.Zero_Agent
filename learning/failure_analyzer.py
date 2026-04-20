"""
失败分析器 - Failure Analyzer
深度分析Agent的失败案例，识别失败模式、根本原因和改进机会
为持续学习和策略优化提供关键洞察
专为CPU环境优化，提供高效的失败诊断和建议生成
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


class FailureType(Enum):
    """失败类型"""
    TOOL_EXECUTION = "tool_execution"  # 工具执行失败
    DECISION_ERROR = "decision_error"  # 决策错误
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # 资源耗尽
    KNOWLEDGE_GAP = "knowledge_gap"  # 知识缺口
    MISUNDERSTANDING = "misunderstanding"  # 理解错误
    TIMEOUT = "timeout"  # 超时
    VALIDATION_FAILED = "validation_failed"  # 验证失败
    UNEXPECTED_ERROR = "unexpected_error"  # 意外错误


class FailureSeverity(Enum):
    """失败严重程度"""
    LOW = "low"  # 低：可自动恢复
    MEDIUM = "medium"  # 中：需要调整策略
    HIGH = "high"  # 高：需要人工干预
    CRITICAL = "critical"  # 严重：系统级问题


class RootCauseCategory(Enum):
    """根本原因类别"""
    INPUT_QUALITY = "input_quality"  # 输入质量问题
    MODEL_LIMITATION = "model_limitation"  # 模型能力限制
    CONFIGURATION = "configuration"  # 配置问题
    EXTERNAL_DEPENDENCY = "external_dependency"  # 外部依赖问题
    LOGIC_ERROR = "logic_error"  # 逻辑错误
    RESOURCE_CONSTRAINT = "resource_constraint"  # 资源约束
    TIMING_ISSUE = "timing_issue"  # 时序问题


@dataclass
class FailureCase:
    """失败案例记录"""
    case_id: str
    failure_type: FailureType
    severity: FailureSeverity
    timestamp: datetime = field(default_factory=datetime.now)

    # 上下文信息
    user_input: str = ""
    cognitive_state: Dict[str, Any] = field(default_factory=dict)
    decision_made: Dict[str, Any] = field(default_factory=dict)
    execution_result: Dict[str, Any] = field(default_factory=dict)

    # 错误信息
    error_message: str = ""
    error_traceback: str = ""
    exception_type: str = ""

    # 分析结果
    root_cause: Optional[str] = None
    root_cause_category: Optional[RootCauseCategory] = None
    contributing_factors: List[str] = field(default_factory=list)
    confidence_at_failure: float = 0.0

    # 改进建议
    suggested_fixes: List[str] = field(default_factory=list)
    prevention_strategies: List[str] = field(default_factory=list)

    # 元数据
    session_id: str = ""
    tags: List[str] = field(default_factory=list)
    analyzed: bool = False
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "case_id": self.case_id,
            "failure_type": self.failure_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "user_input": self.user_input[:200] if self.user_input else "",
            "cognitive_state": self._simplify_dict(self.cognitive_state),
            "decision_made": self._simplify_dict(self.decision_made),
            "execution_result": self._simplify_dict(self.execution_result),
            "error_message": self.error_message[:500],
            "exception_type": self.exception_type,
            "root_cause": self.root_cause,
            "root_cause_category": (
                self.root_cause_category.value
                if self.root_cause_category else None
            ),
            "contributing_factors": self.contributing_factors,
            "confidence_at_failure": round(self.confidence_at_failure, 3),
            "suggested_fixes": self.suggested_fixes,
            "prevention_strategies": self.prevention_strategies,
            "session_id": self.session_id,
            "tags": self.tags,
            "analyzed": self.analyzed,
            "resolved": self.resolved
        }

    @staticmethod
    def _simplify_dict(d: Dict[str, Any], max_depth: int = 2) -> Dict[str, Any]:
        """简化字典"""
        if max_depth <= 0 or not d:
            return {"_simplified": True}

        simplified = {}
        for key, value in d.items():
            if isinstance(value, dict):
                simplified[key] = FailureCase._simplify_dict(value, max_depth - 1)
            elif isinstance(value, (str, int, float, bool)):
                if isinstance(value, str) and len(value) > 200:
                    simplified[key] = value[:200] + "..."
                else:
                    simplified[key] = value
            elif isinstance(value, list):
                simplified[key] = value[:10] if len(value) > 10 else value
            else:
                simplified[key] = str(value)[:100]

        return simplified


@dataclass
@dataclass
class FailurePattern:
    """失败模式"""
    pattern_id: str
    pattern_name: str
    failure_type: FailureType
    occurrence_count: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None

    # 模式特征
    common_characteristics: List[str] = field(default_factory=list)
    typical_scenarios: List[str] = field(default_factory=list)
    affected_components: List[str] = field(default_factory=list)

    # 影响评估
    average_severity: float = 0.0
    impact_score: float = 0.0

    # 解决状态
    resolution_rate: float = 0.0
    known_solutions: List[str] = field(default_factory=list)

    def update_occurrence(self, severity: FailureSeverity, timestamp: datetime):
        """更新出现记录"""
        self.occurrence_count += 1

        if not self.first_seen or timestamp < self.first_seen:
            self.first_seen = timestamp

        if not self.last_seen or timestamp > self.last_seen:
            self.last_seen = timestamp

        # 更新平均严重程度
        severity_map = {
            "low": 0.25,
            "medium": 0.5,
            "high": 0.75,
            "critical": 1.0
        }
        severity_value = severity_map.get(severity.value, 0.5)

        if self.occurrence_count == 1:
            self.average_severity = severity_value
        else:
            self.average_severity = (
                (self.average_severity * (self.occurrence_count - 1) + severity_value)
                / self.occurrence_count
            )

        # 更新影响评分
        self.impact_score = (
            self.occurrence_count * 0.3 +
            self.average_severity * 0.5 +
            (1.0 - self.resolution_rate) * 0.2
        )

@dataclass
class AnalysisConfig:
    """分析配置"""
    min_cases_for_pattern: int = 3  # 形成模式的最小案例数
    analysis_window_hours: int = 24  # 分析时间窗口（小时）
    max_stored_cases: int = 500  # 最大存储案例数

    # 自动分析
    auto_analysis_enabled: bool = True  # 启用自动分析
    analysis_interval_seconds: int = 300  # 分析间隔（秒）

    # 告警阈值
    critical_failure_threshold: int = 5  # 严重失败告警阈值
    failure_rate_threshold: float = 0.2  # 失败率告警阈值

    # 持久化
    persistence_enabled: bool = True
    storage_path: str = "data/failure_analysis"
    save_interval: int = 20  # 保存间隔


class FailureAnalyzer:
    """
    失败分析器

    功能：
    - 失败记录：系统化记录所有失败案例
    - 根因分析：识别失败的根本原因
    - 模式识别：发现重复出现的失败模式
    - 趋势分析：监控失败趋势和变化
    - 建议生成：提供针对性的改进建议
    - 告警机制：在失败率异常时发出告警
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        初始化失败分析器

        Args:
            config: 分析配置
        """
        self.config = config or AnalysisConfig()
        self.logger = logging.getLogger("FailureAnalyzer")

        # 失败案例库
        self.failure_cases: deque = deque(maxlen=self.config.max_stored_cases)
        self.case_index: Dict[str, FailureCase] = {}  # case_id -> case

        # 失败模式库
        self.failure_patterns: Dict[str, FailurePattern] = {}

        # 统计信息
        self.stats = {
            "total_failures": 0,
            "failures_by_type": defaultdict(int),
            "failures_by_severity": defaultdict(int),
            "analyzed_cases": 0,
            "resolved_cases": 0,
            "average_resolution_time": 0.0,
            "current_failure_rate": 0.0
        }

        # 时间窗口统计
        self.recent_interactions: deque = deque(maxlen=1000)
        self.recent_failures: deque = deque(maxlen=200)

        # 上次分析时间
        self.last_analysis_time: Optional[datetime] = None

        # 告警历史
        self.alert_history: List[Dict[str, Any]] = []

        # 创建存储目录
        if self.config.persistence_enabled:
            os.makedirs(self.config.storage_path, exist_ok=True)

        # 加载已有数据
        if self.config.persistence_enabled:
            self._load_from_disk()

        self.logger.info(f"✅ 失败分析器初始化完成")

    def record_failure(
        self,
        error: Exception,
        context: Dict[str, Any],
        decision: Optional[Dict[str, Any]] = None,
        execution_result: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        记录失败案例

        Args:
            error: 异常对象
            context: 失败时的上下文
            decision: 做出的决策
            execution_result: 执行结果

        Returns:
            案例ID
        """
        import uuid

        # 生成唯一ID
        case_id = f"failure_{uuid.uuid4().hex[:8]}"

        # 确定失败类型
        failure_type = self._classify_failure_type(error, context)

        # 确定严重程度
        severity = self._assess_severity(error, context)

        # 提取置信度
        confidence = 0.0
        if context.get('cognitive_state'):
            confidence = context['cognitive_state'].get('confidence_score', 0.0)

        # 创建失败案例
        case = FailureCase(
            case_id=case_id,
            failure_type=failure_type,
            severity=severity,
            user_input=context.get('user_input', ''),
            cognitive_state=context.get('cognitive_state', {}),
            decision_made=decision or {},
            execution_result=execution_result or {},
            error_message=str(error),
            error_traceback=getattr(error, '__traceback__', None),
            exception_type=type(error).__name__,
            confidence_at_failure=confidence,
            session_id=context.get('session_id', ''),
            tags=self._generate_tags(failure_type, context)
        )

        # 存储案例
        self.failure_cases.append(case)
        self.case_index[case_id] = case

        # 更新统计
        self.stats["total_failures"] += 1
        self.stats["failures_by_type"][failure_type.value] += 1
        self.stats["failures_by_severity"][severity.value] += 1

        # 记录到时间窗口
        now = datetime.now()
        self.recent_failures.append({
            "timestamp": now,
            "failure_type": failure_type.value,
            "severity": severity.value
        })

        self.logger.warning(
            f"⚠️ 失败案例已记录: {case_id} - "
            f"类型: {failure_type.value}, "
            f"严重程度: {severity.value}"
        )

        # 检查是否需要告警
        self._check_alert_conditions()

        # 自动分析
        if self.config.auto_analysis_enabled:
            self.analyze_case(case_id)

        return case_id

    def record_interaction(self, success: bool, context: Dict[str, Any]):
        """
        记录交互（成功或失败）用于计算失败率

        Args:
            success: 是否成功
            context: 交互上下文
        """
        now = datetime.now()
        self.recent_interactions.append({
            "timestamp": now,
            "success": success
        })

        # 更新失败率
        self._update_failure_rate()

    def analyze_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """
        分析单个失败案例

        Args:
            case_id: 案例ID

        Returns:
            分析结果
        """
        if case_id not in self.case_index:
            self.logger.warning(f"⚠️ 未找到案例: {case_id}")
            return None

        case = self.case_index[case_id]

        if case.analyzed:
            self.logger.debug(f"ℹ️ 案例已分析过: {case_id}")
            return case.to_dict()

        try:
            self.logger.debug(f"🔍 分析案例: {case_id}")

            # 1. 根因分析
            root_cause, root_cause_category = self._perform_root_cause_analysis(case)
            case.root_cause = root_cause
            case.root_cause_category = root_cause_category

            # 2. 识别贡献因素
            contributing_factors = self._identify_contributing_factors(case)
            case.contributing_factors = contributing_factors

            # 3. 生成修复建议
            suggested_fixes = self._generate_fix_suggestions(case)
            case.suggested_fixes = suggested_fixes

            # 4. 生成预防策略
            prevention_strategies = self._generate_prevention_strategies(case)
            case.prevention_strategies = prevention_strategies

            # 标记为已分析
            case.analyzed = True
            self.stats["analyzed_cases"] += 1

            # 更新模式识别
            self._update_patterns(case)

            self.logger.info(f"✨ 案例分析完成: {case_id}")
            return case.to_dict()

        except Exception as e:
            self.logger.error(f"❌ 案例分析失败: {case_id} - {str(e)}")
            return None

    def analyze_trends(self) -> Dict[str, Any]:
        """
        分析失败趋势

        Returns:
            趋势分析结果
        """
        if len(self.failure_cases) < 5:
            return {"status": "insufficient_data"}

        now = datetime.now()
        window_start = now - timedelta(hours=self.config.analysis_window_hours)

        # 获取时间窗口内的案例
        recent_cases = [
            case for case in self.failure_cases
            if case.timestamp >= window_start
        ]

        if not recent_cases:
            return {
                "status": "no_recent_failures",
                "window_hours": self.config.analysis_window_hours
            }

        # 按时间分组
        hourly_distribution = defaultdict(int)
        for case in recent_cases:
            hour_key = case.timestamp.strftime("%Y-%m-%d %H:00")
            hourly_distribution[hour_key] += 1

        # 计算趋势
        sorted_hours = sorted(hourly_distribution.keys())
        if len(sorted_hours) >= 2:
            first_half = sum(hourly_distribution[h] for h in sorted_hours[:len(sorted_hours)//2])
            second_half = sum(hourly_distribution[h] for h in sorted_hours[len(sorted_hours)//2:])

            if second_half > first_half * 1.2:
                trend = "increasing"
            elif second_half < first_half * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        # 最常见的失败类型
        type_distribution = defaultdict(int)
        for case in recent_cases:
            type_distribution[case.failure_type.value] += 1

        top_types = sorted(type_distribution.items(), key=lambda x: x[1], reverse=True)[:5]

        # 严重程度分布
        severity_distribution = defaultdict(int)
        for case in recent_cases:
            severity_distribution[case.severity.value] += 1

        analysis_result = {
            "status": "completed",
            "analysis_window_hours": self.config.analysis_window_hours,
            "total_recent_failures": len(recent_cases),
            "trend": trend,
            "hourly_distribution": dict(hourly_distribution),
            "top_failure_types": [
                {"type": t, "count": c} for t, c in top_types
            ],
            "severity_distribution": dict(severity_distribution),
            "current_failure_rate": self.stats["current_failure_rate"],
            "timestamp": now.isoformat()
        }

        self.last_analysis_time = now

        self.logger.info(
            f"📊 趋势分析完成 - "
            f"最近{self.config.analysis_window_hours}小时内 {len(recent_cases)} 次失败, "
            f"趋势: {trend}"
        )

        return analysis_result

    def get_improvement_recommendations(self) -> List[Dict[str, Any]]:
        """
        获取改进建议

        Returns:
            改进建议列表
        """
        recommendations = []

        # 基于失败模式
        for pattern in self.failure_patterns.values():
            if pattern.occurrence_count >= self.config.min_cases_for_pattern:
                recommendation = {
                    "priority": "high" if pattern.impact_score > 0.7 else "medium",
                    "category": "pattern_based",
                    "pattern_name": pattern.pattern_name,
                    "description": (
                        f"失败模式 '{pattern.pattern_name}' 已出现 {pattern.occurrence_count} 次"
                    ),
                    "suggested_actions": pattern.known_solutions[:3],
                    "expected_impact": pattern.impact_score,
                    "affected_component": (
                        pattern.affected_components[0]
                        if pattern.affected_components else "unknown"
                    )
                }
                recommendations.append(recommendation)

        # 基于失败率
        if self.stats["current_failure_rate"] > self.config.failure_rate_threshold:
            recommendations.append({
                "priority": "critical",
                "category": "failure_rate",
                "description": (
                    f"当前失败率 {self.stats['current_failure_rate']:.2%} "
                    f"超过阈值 {self.config.failure_rate_threshold:.2%}"
                ),
                "suggested_actions": [
                    "立即审查最近的失败案例",
                    "检查系统配置和资源状态",
                    "考虑临时降低服务负载"
                ],
                "expected_impact": 0.9,
                "affected_component": "system_wide"
            })

        # 基于常见失败类型
        if self.stats["failures_by_type"]:
            top_failure_type = max(
                self.stats["failures_by_type"].items(),
                key=lambda x: x[1]
            )

            if top_failure_type[1] > 10:
                recommendations.append({
                    "priority": "high",
                    "category": "failure_type",
                    "description": (
                        f"失败类型 '{top_failure_type[0]}' 出现频率最高 "
                        f"({top_failure_type[1]} 次)"
                    ),
                    "suggested_actions": self._get_type_specific_recommendations(
                        top_failure_type[0]
                    ),
                    "expected_impact": 0.7,
                    "affected_component": top_failure_type[0]
                })

        # 排序
        priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        recommendations.sort(
            key=lambda r: priority_order.get(r["priority"], 0),
            reverse=True
        )

        return recommendations[:10]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "total_cases_stored": len(self.failure_cases),
            "pattern_count": len(self.failure_patterns),
            "current_failure_rate": self.stats["current_failure_rate"],
            "last_analysis_time": (
                self.last_analysis_time.isoformat()
                if self.last_analysis_time else None
            ),
            "alert_count": len(self.alert_history)
        }

    def mark_resolved(self, case_id: str, resolution_note: str = "") -> bool:
        """
        标记案例已解决

        Args:
            case_id: 案例ID
            resolution_note: 解决说明

        Returns:
            是否成功标记
        """
        if case_id not in self.case_index:
            return False

        case = self.case_index[case_id]
        case.resolved = True
        case.tags.append(f"resolved:{datetime.now().strftime('%Y%m%d')}")

        if resolution_note:
            case.tags.append(f"note:{resolution_note[:50]}")

        self.stats["resolved_cases"] += 1

        # 更新模式的解决率
        self._update_pattern_resolution_rate(case)

        self.logger.info(f"✅ 案例已标记为已解决: {case_id}")
        return True

    def _classify_failure_type(self, error: Exception, context: Dict[str, Any]) -> FailureType:
        """分类失败类型"""
        error_msg = str(error).lower()
        exception_type = type(error).__name__.lower()

        # 工具执行失败
        if any(kw in error_msg for kw in ["tool", "execution", "invoke"]):
            return FailureType.TOOL_EXECUTION

        # 超时
        if any(kw in error_msg for kw in ["timeout", "timed out"]):
            return FailureType.TIMEOUT

        # 资源耗尽
        if any(kw in error_msg for kw in ["memory", "resource", "exhausted"]):
            return FailureType.RESOURCE_EXHAUSTION

        # 验证失败
        if any(kw in error_msg for kw in ["validation", "invalid", "format"]):
            return FailureType.VALIDATION_FAILED

        # 理解决策错误
        if "decision" in exception_type or "cognitive" in exception_type:
            return FailureType.DECISION_ERROR

        # 知识缺口
        if any(kw in error_msg for kw in ["knowledge", "not found", "unknown"]):
            return FailureType.KNOWLEDGE_GAP

        # 默认为意外错误
        return FailureType.UNEXPECTED_ERROR

    def _assess_severity(self, error: Exception, context: Dict[str, Any]) -> FailureSeverity:
        """评估严重程度"""
        error_msg = str(error)

        # 检查是否是关键错误
        critical_indicators = ["critical", "fatal", "system", "crash"]
        if any(indicator in error_msg.lower() for indicator in critical_indicators):
            return FailureSeverity.CRITICAL

        # 检查上下文中的风险等级
        cognitive_state = context.get('cognitive_state', {})
        risk_level = cognitive_state.get('risk_level', 'low')

        if risk_level in ['high', 'critical']:
            return FailureSeverity.HIGH

        # 基于错误类型
        if isinstance(error, (MemoryError, ResourceWarning)):
            return FailureSeverity.HIGH
        elif isinstance(error, (TimeoutError, ConnectionError)):
            return FailureSeverity.MEDIUM
        else:
            return FailureSeverity.LOW

    def _generate_tags(self, failure_type: FailureType, context: Dict[str, Any]) -> List[str]:
        """生成标签"""
        tags = [f"type:{failure_type.value}"]

        # 添加会话标签
        session_id = context.get('session_id', '')
        if session_id:
            tags.append(f"session:{session_id}")

        # 添加用户意图标签
        user_intent = context.get('user_intent', {})
        if isinstance(user_intent, dict):
            intent_type = user_intent.get('intent_type', 'unknown')
            tags.append(f"intent:{intent_type}")

        return tags

    def _perform_root_cause_analysis(
        self,
        case: FailureCase
    ) -> Tuple[Optional[str], Optional[RootCauseCategory]]:
        """执行根因分析"""
        error_msg = case.error_message.lower()
        exception_type = case.exception_type.lower()

        # 输入质量问题
        if case.user_input and len(case.user_input.strip()) == 0:
            return ("用户输入为空", RootCauseCategory.INPUT_QUALITY)

        if any(kw in error_msg for kw in ["parse", "syntax", "format"]):
            return ("输入格式错误", RootCauseCategory.INPUT_QUALITY)

        # 模型能力限制
        if case.confidence_at_failure < 0.3:
            return ("置信度过低，模型能力不足", RootCauseCategory.MODEL_LIMITATION)

        if any(kw in error_msg for kw in ["not implemented", "unsupported"]):
            return ("功能未实现或不支持", RootCauseCategory.MODEL_LIMITATION)

        # 配置问题
        if any(kw in error_msg for kw in ["config", "configuration", "missing"]):
            return ("配置缺失或错误", RootCauseCategory.CONFIGURATION)

        # 外部依赖问题
        if any(kw in error_msg for kw in ["connection", "network", "api", "service"]):
            return ("外部服务不可用", RootCauseCategory.EXTERNAL_DEPENDENCY)

        # 逻辑错误
        if any(kw in exception_type for kw in ["logic", "assertion", "value"]):
            return ("逻辑错误或断言失败", RootCauseCategory.LOGIC_ERROR)

        # 资源约束
        if any(kw in error_msg for kw in ["memory", "cpu", "disk", "quota"]):
            return ("资源不足", RootCauseCategory.RESOURCE_CONSTRAINT)

        # 时序问题
        if any(kw in error_msg for kw in ["race", "concurrent", "async"]):
            return ("并发或时序问题", RootCauseCategory.TIMING_ISSUE)

        # 默认
        return ("未知原因，需要进一步调查", None)

    def _identify_contributing_factors(self, case: FailureCase) -> List[str]:
        """识别贡献因素"""
        factors = []

        # 检查置信度
        if case.confidence_at_failure < 0.4:
            factors.append("决策置信度低")

        # 检查任务复杂度
        cognitive_state = case.cognitive_state
        if cognitive_state:
            complexity = cognitive_state.get('task_complexity', 0.0)
            if complexity > 0.7:
                factors.append("任务复杂度高")

            resource_adequacy = cognitive_state.get('resource_adequacy', 1.0)
            if resource_adequacy < 0.4:
                factors.append("资源充足度低")

        # 检查工具调用
        if case.decision_made:
            tool_calls = case.decision_made.get('tool_calls', [])
            if len(tool_calls) > 3:
                factors.append("工具调用链过长")

        # 检查输入长度
        if case.user_input and len(case.user_input) > 500:
            factors.append("输入过长")

        return factors

    def _generate_fix_suggestions(self, case: FailureCase) -> List[str]:
        """生成修复建议"""
        suggestions = []

        # 基于失败类型
        if case.failure_type == FailureType.TOOL_EXECUTION:
            suggestions.extend([
                "检查工具参数是否正确",
                "添加工具调用的重试机制",
                "验证工具返回结果的格式"
            ])
        elif case.failure_type == FailureType.TIMEOUT:
            suggestions.extend([
                "增加超时时间限制",
                "优化执行流程减少耗时",
                "实现异步执行机制"
            ])
        elif case.failure_type == FailureType.RESOURCE_EXHAUSTION:
            suggestions.extend([
                "优化内存使用",
                "实现资源池管理",
                "添加资源监控和预警"
            ])
        elif case.failure_type == FailureType.KNOWLEDGE_GAP:
            suggestions.extend([
                "扩展知识库",
                "添加澄清询问机制",
                "提供备选方案"
            ])
        elif case.failure_type == FailureType.DECISION_ERROR:
            suggestions.extend([
                "优化决策算法",
                "增加决策验证步骤",
                "引入多策略投票机制"
            ])

        # 基于根因
        if case.root_cause_category == RootCauseCategory.INPUT_QUALITY:
            suggestions.append("增强输入验证和预处理")
        elif case.root_cause_category == RootCauseCategory.EXTERNAL_DEPENDENCY:
            suggestions.append("添加降级和容错机制")

        # 通用建议
        if not suggestions:
            suggestions.append("详细记录错误日志以便进一步分析")

        return suggestions[:5]

    def _generate_prevention_strategies(self, case: FailureCase) -> List[str]:
        """生成预防策略"""
        strategies = []

        # 基于失败类型
        if case.failure_type == FailureType.TOOL_EXECUTION:
            strategies.append("建立工具健康检查机制")
        elif case.failure_type == FailureType.RESOURCE_EXHAUSTION:
            strategies.append("实施资源配额管理")
        elif case.failure_type == FailureType.TIMEOUT:
            strategies.append("设置合理的超时阈值")

        # 基于贡献因素
        if "决策置信度低" in case.contributing_factors:
            strategies.append("提高决策置信度阈值")

        if "任务复杂度高" in case.contributing_factors:
            strategies.append("分解复杂任务为子任务")

        if "资源充足度低" in case.contributing_factors:
            strategies.append("在执行前检查资源可用性")

        # 通用策略
        strategies.append("加强监控和告警")
        strategies.append("定期回顾和优化")

        return list(set(strategies))[:5]

    def _update_patterns(self, case: FailureCase):
        """更新失败模式"""
        # 基于失败类型和根因创建模式键
        pattern_key = f"{case.failure_type.value}_{case.root_cause_category.value if case.root_cause_category else 'unknown'}"

        if pattern_key not in self.failure_patterns:
            # 创建新模式
            pattern = FailurePattern(
                pattern_id=f"pattern_{len(self.failure_patterns) + 1}",
                pattern_name=f"{case.failure_type.value} - {case.root_cause or 'Unknown'}",
                failure_type=case.failure_type,
                common_characteristics=case.contributing_factors[:3],
                affected_components=[case.failure_type.value]
            )
            self.failure_patterns[pattern_key] = pattern

        # 更新模式
        pattern = self.failure_patterns[pattern_key]
        pattern.update_occurrence(case.severity, case.timestamp)

        # 更新已知解决方案
        if case.suggested_fixes:
            for fix in case.suggested_fixes:
                if fix not in pattern.known_solutions:
                    pattern.known_solutions.append(fix)

        self.logger.debug(f"🔄 模式已更新: {pattern_key}")

    def _update_pattern_resolution_rate(self, case: FailureCase):
        """更新模式解决率"""
        pattern_key = f"{case.failure_type.value}_{case.root_cause_category.value if case.root_cause_category else 'unknown'}"

        if pattern_key in self.failure_patterns:
            pattern = self.failure_patterns[pattern_key]
            if pattern.occurrence_count > 0:
                # 这里简化处理，实际应该跟踪每个案例的解决状态
                pass

    def _check_alert_conditions(self):
        """检查告警条件"""
        now = datetime.now()

        # 检查严重失败数量
        critical_count = sum(
            1 for f in self.recent_failures
            if f["severity"] in ["high", "critical"]
            and (now - f["timestamp"]).total_seconds() < 3600
        )

        if critical_count >= self.config.critical_failure_threshold:
            alert = {
                "type": "critical_failure_spike",
                "message": f"过去1小时内发生 {critical_count} 次严重失败",
                "timestamp": now.isoformat(),
                "severity": "critical"
            }
            self.alert_history.append(alert)
            self.logger.critical(f"🚨 告警: {alert['message']}")

        # 检查失败率
        if self.stats["current_failure_rate"] > self.config.failure_rate_threshold:
            alert = {
                "type": "high_failure_rate",
                "message": (
                    f"当前失败率 {self.stats['current_failure_rate']:.2%} "
                    f"超过阈值 {self.config.failure_rate_threshold:.2%}"
                ),
                "timestamp": now.isoformat(),
                "severity": "high"
            }
            self.alert_history.append(alert)
            self.logger.warning(f"⚠️ 告警: {alert['message']}")

    def _update_failure_rate(self):
        """更新失败率"""
        if len(self.recent_interactions) < 10:
            return

        # 计算最近100次交互的失败率
        recent = list(self.recent_interactions)[-100:]
        failures = sum(1 for i in recent if not i["success"])
        self.stats["current_failure_rate"] = failures / len(recent)

    def _get_type_specific_recommendations(self, failure_type: str) -> List[str]:
        """获取特定失败类型的建议"""
        recommendations_map = {
            "tool_execution": [
                "审查工具注册和配置",
                "增强工具调用的错误处理",
                "添加工具性能监控"
            ],
            "timeout": [
                "优化执行效率",
                "实现分步执行机制",
                "增加进度反馈"
            ],
            "resource_exhaustion": [
                "优化资源管理策略",
                "实现垃圾回收机制",
                "添加资源使用预警"
            ],
            "knowledge_gap": [
                "扩展知识库覆盖范围",
                "改善知识检索算法",
                "建立知识更新机制"
            ],
            "decision_error": [
                "优化决策算法",
                "增加决策透明度",
                "引入人工审核环节"
            ]
        }

        return recommendations_map.get(failure_type, ["进一步分析失败原因"])

    def _save_to_disk(self):
        """保存到磁盘"""
        try:
            filepath = os.path.join(
                self.config.storage_path,
                f"failure_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            data = {
                "failure_cases": [case.to_dict() for case in self.failure_cases[-100:]],
                "failure_patterns": {
                    key: {
                        "pattern_id": pattern.pattern_id,
                        "pattern_name": pattern.pattern_name,
                        "failure_type": pattern.failure_type.value,
                        "occurrence_count": pattern.occurrence_count,
                        "average_severity": pattern.average_severity,
                        "impact_score": pattern.impact_score,
                        "known_solutions": pattern.known_solutions
                    }
                    for key, pattern in self.failure_patterns.items()
                },
                "stats": dict(self.stats),
                "timestamp": datetime.now().isoformat()
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.logger.debug(f"💾 失败分析数据已保存: {filepath}")

            # 清理旧文件
            self._cleanup_old_files()

        except Exception as e:
            self.logger.error(f"❌ 保存失败分析数据失败: {str(e)}")

    def _load_from_disk(self):
        """从磁盘加载"""
        try:
            if not os.path.exists(self.config.storage_path):
                return

            files = [
                f for f in os.listdir(self.config.storage_path)
                if f.startswith("failure_analysis_") and f.endswith(".json")
            ]

            if not files:
                return

            latest_file = sorted(files)[-1]
            filepath = os.path.join(self.config.storage_path, latest_file)

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 恢复统计数据
            if "stats" in data:
                self.stats.update(data["stats"])

            self.logger.info(f"📂 已加载失败分析数据")

        except Exception as e:
            self.logger.error(f"❌ 加载失败分析数据失败: {str(e)}")

    def _cleanup_old_files(self, keep_count: int = 5):
        """清理旧文件"""
        try:
            files = [
                f for f in os.listdir(self.config.storage_path)
                if f.startswith("failure_analysis_") and f.endswith(".json")
            ]

            if len(files) > keep_count:
                sorted_files = sorted(files)
                for old_file in sorted_files[:-keep_count]:
                    filepath = os.path.join(self.config.storage_path, old_file)
                    os.remove(filepath)
                    self.logger.debug(f"🗑️ 已删除旧文件: {old_file}")

        except Exception as e:
            self.logger.error(f"❌ 清理旧文件失败: {str(e)}")
