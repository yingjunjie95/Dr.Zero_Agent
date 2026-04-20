"""
决策制定器 - Decision Maker
负责将认知阶段的策略转化为具体的行动计划
协调工具执行、风险评估和动态调整
实现自主、安全、高效的决策能力
"""
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class DecisionType(Enum):
    """决策类型"""
    DIRECT_RESPONSE = "direct_response"  # 直接响应
    TOOL_EXECUTION = "tool_execution"  # 工具执行
    CLARIFICATION = "clarification"  # 澄清询问
    ESCALATION = "escalation"  # 升级处理
    DEFERRED = "deferred"  # 延迟决策


class ExecutionStatus(Enum):
    """执行状态"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_CONFIRMATION = "waiting_confirmation"


@dataclass
class ActionPlan:
    """行动计划"""
    action_id: str
    decision_type: DecisionType
    primary_action: str
    tool_calls: List[Dict[str, Any]]
    memory_operations: List[Dict[str, Any]]
    fallback_actions: List[str]
    confidence_threshold: float
    max_execution_time: float
    requires_confirmation: bool

    execution_status: ExecutionStatus = ExecutionStatus.PENDING
    execution_result: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "action_id": self.action_id,
            "decision_type": self.decision_type.value,
            "primary_action": self.primary_action,
            "tool_calls": self.tool_calls,
            "memory_operations": self.memory_operations,
            "fallback_actions": self.fallback_actions,
            "confidence_threshold": self.confidence_threshold,
            "max_execution_time": self.max_execution_time,
            "requires_confirmation": self.requires_confirmation,
            "execution_status": self.execution_status.value,
            "execution_result": self.execution_result,
            "execution_time": round(self.execution_time, 3),
            "created_at": self.created_at.isoformat()
        }


@dataclass
class DecisionOutcome:
    """决策结果"""
    success: bool
    response: str
    action_plan: ActionPlan
    execution_metrics: Dict[str, Any]
    confidence_score: float
    reasoning_trace: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "response": self.response,
            "action_plan": self.action_plan.to_dict(),
            "execution_metrics": self.execution_metrics,
            "confidence_score": round(self.confidence_score, 3),
            "reasoning_trace": self.reasoning_trace,
            "timestamp": self.timestamp.isoformat()
        }


class DecisionMaker:
    """
    决策制定器

    功能：
    - 策略转化：将认知策略转化为具体行动计划
    - 工具协调：调度和执行所需工具
    - 风险管理：处理高风险决策的用户确认
    - 动态调整：根据执行情况调整决策
    - 结果评估：评估决策效果和置信度
    """

    def __init__(self, tool_engine: Any, metacognitive_engine: Any):
        """
        初始化决策制定器

        Args:
            tool_engine: 工具引擎实例
            metacognitive_engine: 元认知引擎实例
        """
        self.tool_engine = tool_engine
        self.metacognitive_engine = metacognitive_engine
        self.logger = logging.getLogger("DecisionMaker")

        # 决策历史
        self.decision_history: List[DecisionOutcome] = []
        self.max_history_size = 100

        # 性能统计
        self.stats = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "failed_decisions": 0,
            "confirmation_requests": 0,
            "avg_execution_time": 0.0
        }

        self.logger.info("✅ 决策制定器初始化完成")

    def make_decision(
        self,
        cognitive_state: Dict[str, Any],
        perception: Dict[str, Any]
    ) -> DecisionOutcome:
        """
        制定决策并执行

        Args:
            cognitive_state: 认知状态（包含策略选择、元认知评估等）
            perception: 感知结果（包含用户输入、环境信息等）

        Returns:
            决策结果
        """
        start_time = time.time()
        reasoning_trace = []

        try:
            self.logger.debug("🧠 开始制定决策...")

            # 1. 提取关键信息
            strategy = cognitive_state.get('selected_strategy', {})
            metacognitive = cognitive_state.get('metacognitive_assessment', {})
            confidence = cognitive_state.get('confidence_score', 0.5)
            risk_level = cognitive_state.get('risk_level', 0.0)

            reasoning_trace.append(f"策略类型: {strategy.get('type', 'unknown')}")
            reasoning_trace.append(f"置信度: {confidence:.2f}")
            reasoning_trace.append(f"风险等级: {risk_level:.2f}")

            # 2. 创建行动计划
            action_plan = self._create_action_plan(
                strategy, metacognitive, perception
            )
            reasoning_trace.append(f"行动计划已创建: {action_plan.action_id}")

            # 3. 检查是否需要用户确认
            if action_plan.requires_confirmation:
                self.stats['confirmation_requests'] += 1
                reasoning_trace.append("需要用户确认")

                confirmation_response = self._request_user_confirmation(
                    action_plan, perception
                )

                execution_time = time.time() - start_time
                action_plan.execution_time = execution_time
                action_plan.execution_status = ExecutionStatus.WAITING_CONFIRMATION

                outcome = DecisionOutcome(
                    success=False,
                    response=confirmation_response,
                    action_plan=action_plan,
                    execution_metrics={'requires_confirmation': True},
                    confidence_score=confidence,
                    reasoning_trace=reasoning_trace
                )

                self._record_decision(outcome)
                return outcome

            # 4. 执行行动计划
            reasoning_trace.append("开始执行行动计划")
            response, execution_metrics = self._execute_action_plan(
                action_plan, perception
            )

            execution_time = time.time() - start_time
            action_plan.execution_time = execution_time

            # 5. 评估执行结果
            success = execution_metrics.get('errors', 0) == 0
            action_plan.execution_status = (
                ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED
            )
            action_plan.execution_result = execution_metrics

            reasoning_trace.append(f"执行完成: {'成功' if success else '失败'}")
            reasoning_trace.append(f"执行时间: {execution_time:.2f}s")

            # 6. 创建决策结果
            outcome = DecisionOutcome(
                success=success,
                response=response,
                action_plan=action_plan,
                execution_metrics=execution_metrics,
                confidence_score=confidence,
                reasoning_trace=reasoning_trace
            )

            # 7. 记录决策
            self._record_decision(outcome)

            self.logger.debug(f"✨ 决策完成 - 成功: {success}, 耗时: {execution_time:.2f}s")
            return outcome

        except Exception as e:
            self.logger.error(f"❌ 决策制定失败: {str(e)}", exc_info=True)

            execution_time = time.time() - start_time
            error_outcome = self._create_error_outcome(
                e, execution_time, reasoning_trace
            )
            self._record_decision(error_outcome)

            return error_outcome

    def _create_action_plan(
        self,
        strategy: Dict[str, Any],
        metacognitive: Dict[str, Any],
        perception: Dict[str, Any]
    ) -> ActionPlan:
        """创建行动计划"""
        import uuid

        # 生成唯一ID
        action_id = f"action_{uuid.uuid4().hex[:8]}"

        # 确定决策类型
        primary_action = strategy.get('primary_action', 'direct_response')
        decision_type = self._map_decision_type(primary_action)

        # 提取工具调用
        tool_calls = strategy.get('tool_calls', [])

        # 提取记忆操作
        memory_operations = strategy.get('memory_operations', [])

        # 获取回退动作
        fallback_actions = strategy.get('fallback_actions', [])

        # 置信度阈值
        confidence_threshold = strategy.get('confidence_threshold', 0.6)

        # 最大执行时间
        max_execution_time = strategy.get('max_execution_time', 30.0)

        # 是否需要确认（基于风险和策略）
        requires_confirmation = strategy.get('requires_confirmation', False)

        # 如果风险等级很高，强制要求确认
        risk_level_str = metacognitive.get('risk_level', 'low')
        if risk_level_str in ['high', 'critical']:
            requires_confirmation = True

        action_plan = ActionPlan(
            action_id=action_id,
            decision_type=decision_type,
            primary_action=primary_action,
            tool_calls=tool_calls,
            memory_operations=memory_operations,
            fallback_actions=fallback_actions,
            confidence_threshold=confidence_threshold,
            max_execution_time=max_execution_time,
            requires_confirmation=requires_confirmation
        )

        self.logger.debug(f"📋 行动计划已创建: {action_id}")
        return action_plan

    def _map_decision_type(self, primary_action: str) -> DecisionType:
        """映射决策类型"""
        type_mapping = {
            'direct_response': DecisionType.DIRECT_RESPONSE,
            'use_tools': DecisionType.TOOL_EXECUTION,
            'clarify': DecisionType.CLARIFICATION,
            'escalate': DecisionType.ESCALATION,
            'defer': DecisionType.DEFERRED
        }
        return type_mapping.get(primary_action, DecisionType.DIRECT_RESPONSE)

    def _execute_action_plan(
        self,
        action_plan: ActionPlan,
        perception: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """执行行动计划"""
        execution_start = time.time()
        execution_metrics = {
            'tool_calls': 0,
            'memory_operations': 0,
            'errors': 0,
            'warnings': 0
        }

        try:
            # 更新状态
            action_plan.execution_status = ExecutionStatus.EXECUTING

            # 根据决策类型执行
            if action_plan.decision_type == DecisionType.DIRECT_RESPONSE:
                response = self._execute_direct_response(action_plan, perception)

            elif action_plan.decision_type == DecisionType.TOOL_EXECUTION:
                response = self._execute_with_tools(action_plan, perception, execution_metrics)

            elif action_plan.decision_type == DecisionType.CLARIFICATION:
                response = self._execute_clarification(action_plan, perception)

            elif action_plan.decision_type == DecisionType.ESCALATION:
                response = self._execute_escalation(action_plan, perception)

            else:
                response = self._execute_default_response(perception)

            # 执行记忆操作
            if action_plan.memory_operations:
                self._execute_memory_operations(action_plan.memory_operations, perception)
                execution_metrics['memory_operations'] = len(action_plan.memory_operations)

            execution_time = time.time() - execution_start
            execution_metrics['execution_time'] = execution_time

            return response, execution_metrics

        except Exception as e:
            execution_metrics['errors'] += 1
            self.logger.error(f"执行行动计划失败: {str(e)}")

            # 尝试执行回退动作
            if action_plan.fallback_actions:
                self.logger.warning("⚠️ 尝试执行回退动作")
                return self._execute_fallback_actions(
                    action_plan, perception, execution_metrics
                )

            # 返回错误响应
            error_response = self._generate_error_response(e, action_plan)
            execution_time = time.time() - execution_start
            execution_metrics['execution_time'] = execution_time

            return error_response, execution_metrics

    def _execute_direct_response(
        self,
        action_plan: ActionPlan,
        perception: Dict[str, Any]
    ) -> str:
        """执行直接响应"""
        user_input = perception.get('user_input', '')
        context = perception.get('current_context', {})

        # 这里应该调用语言模型生成响应
        # 由于是CPU环境，使用简化的响应生成
        response = self._generate_contextual_response(user_input, context)

        self.logger.debug("✅ 直接响应生成完成")
        return response

    def _execute_with_tools(
        self,
        action_plan: ActionPlan,
        perception: Dict[str, Any],
        execution_metrics: Dict[str, Any]
    ) -> str:
        """执行工具调用"""
        user_input = perception.get('user_input', '')
        responses = []

        # 执行工具调用
        for tool_call in action_plan.tool_calls:
            tool_name = tool_call.get('name', '')
            tool_params = tool_call.get('parameters', {})

            try:
                self.logger.debug(f"🔧 调用工具: {tool_name}")

                result = self.tool_engine.execute_tool(
                    tool_name=tool_name,
                    parameters=tool_params,
                    context=perception
                )

                execution_metrics['tool_calls'] += 1

                if result.get('success', False):
                    responses.append(result.get('output', ''))
                else:
                    execution_metrics['warnings'] += 1
                    self.logger.warning(f"⚠️ 工具调用失败: {tool_name} - {result.get('error', '')}")

            except Exception as e:
                execution_metrics['errors'] += 1
                self.logger.error(f"❌ 工具执行异常: {tool_name} - {str(e)}")

        # 整合工具结果
        if responses:
            combined_response = self._synthesize_tool_results(responses, user_input)
        else:
            combined_response = "抱歉，工具执行未能产生有效结果。"

        return combined_response

    def _execute_clarification(
        self,
        action_plan: ActionPlan,
        perception: Dict[str, Any]
    ) -> str:
        """执行澄清询问"""
        user_input = perception.get('user_input', '')

        clarification_questions = [
            "为了更好地帮助您，能否请您详细说明一下？",
            "您的具体需求是什么？",
            "能否提供更多背景信息？"
        ]

        response = (
            f"我注意到您的问题可能需要更多信息才能准确回答。\n\n"
            f"{clarification_questions[0]}\n\n"
            f"您可以从以下角度补充：\n"
            f"- 具体场景或背景\n"
            f"- 期望达到的目标\n"
            f"- 已有的尝试或想法"
        )

        self.logger.debug("✅ 澄清询问生成完成")
        return response

    def _execute_escalation(
        self,
        action_plan: ActionPlan,
        perception: Dict[str, Any]
    ) -> str:
        """执行升级处理"""
        response = (
            "这个问题超出了我的当前能力范围。\n\n"
            "建议您：\n"
            "1. 简化问题或分解为多个小问题\n"
            "2. 查阅相关专业资料\n"
            "3. 咨询领域专家\n\n"
            "我会继续学习，希望未来能更好地帮助您。"
        )

        self.logger.warning("⚠️ 问题已升级处理")
        return response

    def _execute_default_response(self, perception: Dict[str, Any]) -> str:
        """执行默认响应"""
        return "我已收到您的请求，正在处理中..."

    def _execute_fallback_actions(
        self,
        action_plan: ActionPlan,
        perception: Dict[str, Any],
        execution_metrics: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """执行回退动作"""
        self.logger.info("🔄 执行回退动作")

        # 简化为直接响应
        fallback_response = self._execute_direct_response(action_plan, perception)

        execution_metrics['fallback_used'] = True
        return fallback_response, execution_metrics

    def _execute_memory_operations(
        self,
        operations: List[Dict[str, Any]],
        perception: Dict[str, Any]
    ):
        """执行记忆操作"""
        # 这里应该与记忆系统交互
        # 暂时简化处理
        for operation in operations:
            op_type = operation.get('type', '')
            self.logger.debug(f"💾 执行记忆操作: {op_type}")

    def _request_user_confirmation(
        self,
        action_plan: ActionPlan,
        perception: Dict[str, Any]
    ) -> str:
        """请求用户确认"""
        risk_level = "高"
        if action_plan.tool_calls:
            tools_info = ", ".join([tc.get('name', '') for tc in action_plan.tool_calls])
            confirmation_msg = (
                f"⚠️ 此操作涉及以下工具调用：{tools_info}\n\n"
                f"由于风险等级较高，需要您的确认才能继续。\n\n"
                f"请回复「确认」以继续，或「取消」以中止。"
            )
        else:
            confirmation_msg = (
                f"⚠️ 此操作存在一定风险，需要您的确认。\n\n"
                f"请回复「确认」以继续，或「取消」以中止。"
            )

        return confirmation_msg

    def _generate_contextual_response(
        self,
        user_input: str,
        context: Dict[str, Any]
    ) -> str:
        """生成上下文相关的响应"""
        # 这是一个占位符实现
        # 实际应该调用语言模型
        return f"我已理解您的问题：{user_input[:50]}...\n\n基于当前上下文，我将为您提供帮助。"

    def _synthesize_tool_results(
        self,
        results: List[str],
        user_input: str
    ) -> str:
        """综合工具执行结果"""
        if not results:
            return "工具执行未产生结果。"

        synthesized = "根据工具执行结果：\n\n"
        for i, result in enumerate(results, 1):
            synthesized += f"{i}. {result}\n\n"

        synthesized += "\n以上信息仅供参考，如有需要我可以进一步分析。"
        return synthesized

    def _generate_error_response(
        self,
        error: Exception,
        action_plan: ActionPlan
    ) -> str:
        """生成错误响应"""
        return (
            f"抱歉，在处理您的请求时遇到了问题：\n\n"
            f"错误信息：{str(error)}\n\n"
            f"请稍后重试，或尝试用不同的方式描述您的需求。"
        )

    def _create_error_outcome(
        self,
        error: Exception,
        execution_time: float,
        reasoning_trace: List[str]
    ) -> DecisionOutcome:
        """创建错误结果"""
        import uuid

        error_plan = ActionPlan(
            action_id=f"error_{uuid.uuid4().hex[:8]}",
            decision_type=DecisionType.ESCALATION,
            primary_action="error_handling",
            tool_calls=[],
            memory_operations=[],
            fallback_actions=[],
            confidence_threshold=0.0,
            max_execution_time=0.0,
            requires_confirmation=False,
            execution_status=ExecutionStatus.FAILED,
            execution_time=execution_time
        )

        return DecisionOutcome(
            success=False,
            response=self._generate_error_response(error, error_plan),
            action_plan=error_plan,
            execution_metrics={'error': str(error)},
            confidence_score=0.0,
            reasoning_trace=reasoning_trace + [f"错误：{str(error)}"]
        )

    def _record_decision(self, outcome: DecisionOutcome):
        """记录决策"""
        self.decision_history.append(outcome)

        # 限制历史记录大小
        if len(self.decision_history) > self.max_history_size:
            self.decision_history.pop(0)

        # 更新统计
        self.stats['total_decisions'] += 1
        if outcome.success:
            self.stats['successful_decisions'] += 1
        else:
            self.stats['failed_decisions'] += 1

        # 更新平均执行时间
        total_time = sum(d.action_plan.execution_time for d in self.decision_history)
        self.stats['avg_execution_time'] = total_time / len(self.decision_history)

    def get_decision_statistics(self) -> Dict[str, Any]:
        """获取决策统计信息"""
        return {
            **self.stats,
            "history_size": len(self.decision_history),
            "success_rate": (
                self.stats['successful_decisions'] / self.stats['total_decisions']
                if self.stats['total_decisions'] > 0 else 0.0
            )
        }

    def get_recent_decisions(self, count: int = 10) -> List[Dict[str, Any]]:
        """获取最近的决策记录"""
        recent = self.decision_history[-count:]
        return [d.to_dict() for d in reversed(recent)]
