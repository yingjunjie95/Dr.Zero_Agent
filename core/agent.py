"""
Dr.Zero Agent 主类
实现真正的自主智能体，协调感知、认知、记忆、行动和学习模块
在CPU环境下高效运行，体现自主性、反应性、主动性和适应性
"""
import time
import threading
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from .perception.environment_monitor import EnvironmentMonitor
from .perception.intent_analyzer import IntentAnalyzer
from .perception.emotion_detector import EmotionDetector
from .cognition.metacognitive_engine import MetacognitiveEngine
from .cognition.self_model import SelfModel
from .cognition.strategy_selector import StrategySelector
from .memory.memory_system import MemorySystem
from .action.tool_engine import ToolEngine
from .action.decision_maker import DecisionMaker
from learning.continuous_learner import ContinuousLearner
from learning.experience_replay import ExperienceReplay
from learning.strategy_optimizer import StrategyOptimizer
from monitoring.performance_tracker import PerformanceTracker
from config.system_config import SystemConfig
from config.agent_profile import AgentProfile
from tools.registry import ToolRegistry
from tools.document_processor import DocumentProcessor
from tools.api_connector import APIConnector, AuthType


class DrZeroAgent:
    """
    Dr.Zero Agent主类 - 一个具有自主认知能力的智能体
    特性：
    - 自主目标管理和决策
    - 三层记忆架构（感官/工作/长期记忆）
    - 元认知能力（自我评估、风险控制）
    - 动态资源优化
    - 持续学习和适应
    - 情绪感知和响应
    """

    def __init__(self, config: SystemConfig = None, profile: AgentProfile = None):
        """
        初始化Dr.Zero Agent

        Args:
            config: 系统配置，包含硬件参数和性能设置
            profile: Agent特性配置，包含性格、能力和边界
        """
        # 配置初始化
        self.config = config or SystemConfig()
        self.profile = profile or AgentProfile()

        # 设置日志
        self._setup_logging()

        # 核心组件初始化
        self._initialize_core_components()

        # 状态管理
        self.is_active = True
        self.current_goal = None
        self.thought_process = []
        self.session_id = self._generate_session_id()

        # 性能监控
        self.performance_tracker = PerformanceTracker()

        # 启动后台任务
        self._start_background_tasks()

        self.logger.info(f"Dr.Zero Agent 初始化完成 - 会话ID: {self.session_id}")

    def _setup_logging(self):
        """配置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"dr_zero_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("DrZeroAgent")

    def _initialize_core_components(self):
        """初始化所有核心组件"""
        try:
            # 1. 感知模块
            self.environment_monitor = EnvironmentMonitor()
            self.intent_analyzer = IntentAnalyzer(model_name=self.config.MODEL_NAME)
            self.emotion_detector = EmotionDetector()

            # 2. 认知模块
            self.metacognitive_engine = MetacognitiveEngine(
                self_model=SelfModel(agent_profile=self.profile),
                risk_threshold=self.profile.RISK_AVERSION
            )
            self.strategy_selector = StrategySelector()

            # 3. 记忆系统
            self.memory_system = MemorySystem(
                max_sensory_size=100,
                working_memory_capacity=50,
                embedding_dim=768  # 与qwen模型匹配
            )

            # 4. 行动模块
            self.tool_registry = ToolRegistry()
            self._register_default_tools()
            self.tool_engine = ToolEngine(tool_registry=self.tool_registry)
            self.decision_maker = DecisionMaker(
                tool_engine=self.tool_engine,
                metacognitive_engine=self.metacognitive_engine
            )

            # 5. 学习模块
            self.continuous_learner = ContinuousLearner(
                learning_rate=self.profile.LEARNING_RATE,
                exploration_rate=self.profile.EXPLORATION_RATE
            )
            
            # 6. 经验回放和策略优化
            self.experience_replay = ExperienceReplay()
            self.strategy_optimizer = StrategyOptimizer()

            # 7. 工具模块
            self.document_processor = DocumentProcessor()
            self.api_connector = APIConnector()

            self.logger.info("✅ 所有核心组件初始化成功")

        except Exception as e:
            self.logger.error(f"❌ 组件初始化失败: {str(e)}")
            raise

    def _code_executor_tool(self, code: str, **kwargs):
        """代码执行工具（带沙箱保护）"""
        from .action.execution_sandbox import ExecutionSandbox, SandboxConfig, SecurityLevel

        # 创建沙箱实例
        sandbox = ExecutionSandbox(
            config=SandboxConfig(
                max_execution_time=5.0,
                security_level=SecurityLevel.STRICT
            )
        )

        # 执行代码
        result = sandbox.execute_code(code)

        if result.success:
            return {
                "success": True,
                "output": result.output,
                "execution_time": result.execution_time
            }
        else:
            return {
                "success": False,
                "error": result.error,
                "status": result.status.value
            }

    def _web_search_tool(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        网络搜索工具
        
        Args:
            query: 搜索查询
            num_results: 结果数量
            structured_output: 是否结构化输出
            
        Returns:
            搜索结果
        """
        try:
            from tools.web_search import WebSearchTool, SearchEngine
            
            search_tool = WebSearchTool(
                default_engine=SearchEngine.DUCKDUCKGO,
                max_results=kwargs.get('num_results', 5),
                enable_multi_source=True,
                enable_reranking=True
            )
            
            results = search_tool.search(
                query=query,
                num_results=kwargs.get('num_results', 5),
                structured_output=kwargs.get('structured_output', True)
            )
            
            return {
                "success": True,
                "query": query,
                "results": results.get('results', []),
                "count": results.get('result_count', 0),
                "search_metadata": results.get('search_metadata', {}),
                "structured_output": results
            }
        except Exception as e:
            self.logger.error(f"网络搜索失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }

    def _document_analysis_tool(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        文档分析工具
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档分析结果
        """
        try:
            result = self.document_processor.process_document(file_path)
            
            return {
                "success": True,
                "file_path": file_path,
                "content": result.get("content", ""),
                "metadata": result.get("metadata", {})
            }
        except Exception as e:
            self.logger.error(f"文档分析失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }

    def _calculator_tool(self, expression: str, **kwargs) -> Dict[str, Any]:
        """
        计算器工具
        
        Args:
            expression: 数学表达式
            
        Returns:
            计算结果
        """
        try:
            # 安全的表达式评估
            allowed_chars = set('0123456789+-*/().^ ')
            if not all(c in allowed_chars for c in expression):
                raise ValueError("表达式包含不允许的字符")
            
            # 替换 ^ 为 **
            expression = expression.replace('^', '**')
            
            # 安全地计算表达式
            result = eval(expression, {"__builtins__": {}}, {})
            
            return {
                "success": True,
                "expression": expression,
                "result": result
            }
        except Exception as e:
            self.logger.error(f"计算失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "expression": expression
            }

    def _register_default_tools(self):
        """注册默认工具集"""
        # 初始化文档处理器
        self.document_processor = DocumentProcessor()
        
        # 网络搜索工具
        self.tool_registry.register_tool(
            name="web_search",
            function=self._web_search_tool,
            description="安全的网络搜索工具，用于获取最新信息",
            requires_permission=True,
            max_calls_per_session=5,
            category="search"
        )

        # 文档处理工具
        self.tool_registry.register_tool(
            name="document_analysis",
            function=self._document_analysis_tool,
            description="分析和处理文本文档内容",
            requires_permission=False,
            category="document"
        )

        # 计算工具
        self.tool_registry.register_tool(
            name="calculator",
            function=self._calculator_tool,
            description="执行数学计算和逻辑运算",
            requires_permission=False,
            category="calculation"
        )

        # 代码执行工具（受限）
        self.tool_registry.register_tool(
            name="code_executor",
            function=self._code_executor_tool,
            description="在安全沙箱中执行简单Python代码",
            requires_permission=True,
            category="code"
        )

        # API调用工具
        self.tool_registry.register_tool(
            name="api_call",
            function=self._api_call_tool,
            description="调用外部API接口获取数据",
            requires_permission=True,
            max_calls_per_session=10,
            category="api"
        )

        self.logger.info(f"✅ 已注册 {len(self.tool_registry.tools)} 个工具")

    def _api_call_tool(
            self,
            api_name: str,
            endpoint: str,
            method: str = "GET",
            params: Optional[Dict[str, Any]] = None,
            data: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """
        API调用工具

        Args:
            api_name: 注册的API名称
            endpoint: API端点
            method: HTTP方法 (GET/POST/PUT/DELETE)
            params: URL参数
            data: 请求体数据

        Returns:
            API响应结果
        """
        try:
            from tools.api_connector import HttpMethod

            # 转换HTTP方法
            method_map = {
                "GET": HttpMethod.GET,
                "POST": HttpMethod.POST,
                "PUT": HttpMethod.PUT,
                "DELETE": HttpMethod.DELETE,
                "PATCH": HttpMethod.PATCH
            }

            http_method = method_map.get(method.upper(), HttpMethod.GET)

            # 执行API调用
            result = self.api_connector.request(
                url=endpoint,
                method=http_method,
                params=params,
                data=data,
                api_name=api_name
            )

            return result

        except Exception as e:
            self.logger.error(f"API调用失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "status_code": 0,
                "data": None
            }

    def register_custom_api(
            self,
            name: str,
            base_url: str,
            auth_type: str = "none",
            **auth_params
    ):
        """
        注册自定义API

        Args:
            name: API名称
            base_url: 基础URL
            auth_type: 认证类型 (none/api_key/bearer_token/basic_auth)
            **auth_params: 认证参数
        """
        auth_type_map = {
            "none": AuthType.NONE,
            "api_key": AuthType.API_KEY,
            "bearer_token": AuthType.BEARER_TOKEN,
            "basic_auth": AuthType.BASIC_AUTH
        }

        auth_type_enum = auth_type_map.get(auth_type.lower(), AuthType.NONE)

        self.api_connector.register_api(
            name=name,
            base_url=base_url,
            auth_type=auth_type_enum,
            **auth_params
        )

        self.logger.info(f"✅ 已注册自定义API: {name}")

    def _start_background_tasks(self):
        """启动后台维护任务"""
        # 资源监控线程
        self.resource_monitor_thread = threading.Thread(
            target=self._resource_monitoring_loop,
            daemon=True
        )

        # 记忆维护线程
        self.memory_maintenance_thread = threading.Thread(
            target=self._memory_maintenance_loop,
            daemon=True
        )

        # 学习优化线程
        self.learning_thread = threading.Thread(
            target=self._continuous_learning_loop,
            daemon=True
        )

        self.resource_monitor_thread.start()
        self.memory_maintenance_thread.start()
        self.learning_thread.start()

        self.logger.info("🔄 后台维护任务已启动")

    def process_input(self, user_input: str) -> str:
        """
        处理用户输入，生成Agent响应

        Args:
            user_input: 用户输入的文本

        Returns:
            Agent的响应文本
        """
        start_time = time.time()
        interaction_id = self._generate_interaction_id()

        try:
            self.logger.info(f"🔄 处理用户输入 (ID: {interaction_id})")
            self.logger.debug(f"用户输入: {user_input}")

            # 1. 感知阶段 - 收集环境信息和用户意图
            perception = self._perceive_environment(user_input)

            # 2. 认知阶段 - 评估情况和制定策略
            cognitive_state = self._cognitive_processing(perception)

            # 3. 决策阶段 - 选择最佳行动方案
            decision = self._make_decision(cognitive_state, perception)

            # 4. 执行阶段 - 执行决策并获取结果
            response, execution_metrics = self._execute_decision(decision, perception)

            # 5. 学习阶段 - 记录经验和优化策略
            self._learn_from_interaction(
                user_input, response, perception, cognitive_state, decision, execution_metrics
            )

            # 6. 更新状态
            self._update_agent_state(perception, cognitive_state, decision, response)

            processing_time = time.time() - start_time
            self.performance_tracker.record_interaction(
                processing_time, execution_metrics, cognitive_state
            )

            self.logger.info(f"✅ 交互处理完成 (ID: {interaction_id}, 耗时: {processing_time:.2f}s)")
            return response

        except Exception as e:
            self.logger.error(f"❌ 处理输入时出错: {str(e)}")
            return self._handle_error(e, user_input)

    def _handle_error(self, error: Exception, user_input: str) -> str:
        """
        处理错误并生成友好的错误响应
        
        Args:
            error: 发生的异常
            user_input: 用户输入
            
        Returns:
            错误响应文本
        """
        error_msg = str(error)
        self.logger.error(f"错误详情: {type(error).__name__}: {error_msg}")
        
        # 根据错误类型提供不同的响应
        if isinstance(error, AttributeError):
            return f"抱歉，系统内部出现了一些问题（{error_msg}）。我正在记录这个问题以便后续修复。您可以尝试重新描述您的需求。"
        elif isinstance(error, TimeoutError):
            return "处理超时了，可能是请求太复杂。请简化您的问题或稍后重试。"
        elif isinstance(error, MemoryError):
            return "系统内存不足，我需要清理一些缓存。请稍后再试。"
        else:
            return f"处理您的请求时遇到了问题：{error_msg}。我会继续改进我的能力，感谢您的耐心。"

    def _handle_execution_error(self, error: Exception, decision: Dict[str, Any], perception: Dict[str, Any]) -> str:
        """
        处理执行阶段的错误
        
        Args:
            error: 执行错误
            decision: 决策信息
            perception: 感知信息
            
        Returns:
            错误响应文本
        """
        error_msg = str(error)
        self.logger.error(f"执行错误: {type(error).__name__}: {error_msg}")
        
        # 尝试使用备选方案
        action_plan = decision.get('action_plan', {})
        fallback_actions = action_plan.get('fallback_actions', [])
        if fallback_actions:
            self.logger.info(f"🔄 尝试备选方案: {len(fallback_actions)} 个")
            return f"主要方案执行失败，正在尝试备选方案... ({error_msg})"
        
        # 返回友好错误消息
        return f"很抱歉，在执行操作时遇到了问题：{error_msg}。我会记录下来并在未来改进。您能否换个方式描述您的需求？"

    def _generate_default_response(self, perception: Dict[str, Any]) -> str:
        """
        生成默认响应
        
        Args:
            perception: 感知信息
            
        Returns:
            默认响应文本
        """
        user_input = perception.get('user_input', '')
        user_intent = perception.get('user_intent', {})
        user_emotion = perception.get('user_emotion', {})
        
        # 基于意图类型生成响应
        intent_type = user_intent.get('intent_type', 'unknown') if isinstance(user_intent, dict) else getattr(user_intent, 'intent_type', None)
        
        if intent_type:
            intent_value = intent_type.value if hasattr(intent_type, 'value') else str(intent_type)
            
            # 新增：处理问候和称呼
            if intent_value in ['greeting', 'name_mention']:
                # 提取用户提到的名字
                name = user_input.strip()
                return f"你好，{name}！很高兴认识你。我是 Dr.Zero Agent，有什么我可以帮助你的吗？"
            
            if intent_value in ['conversation', 'greeting']:
                return "您好！我是 Dr.Zero Agent，很高兴与您交流。请问有什么我可以帮助您的吗？"
            elif intent_value == 'query':
                return "我理解您想了解一些信息。能否请您更详细地描述您的问题？这样我可以更好地帮助您。"
            elif intent_value == 'clarification':
                return "我需要更多信息来准确回答您的问题。您能提供更多细节吗？"
        
        # 检查用户情绪
        emotion_type = user_emotion.get('primary_emotion', None) if isinstance(user_emotion, dict) else getattr(user_emotion, 'primary_emotion', None)
        if emotion_type:
            emotion_value = emotion_type.value if hasattr(emotion_type, 'value') else str(emotion_type)
            if emotion_value in ['frustration', 'anger']:
                return "我理解这可能让您感到困扰。我会尽力帮助您解决问题。请告诉我具体需要什么帮助。"
        
        # 通用默认响应
        return f"我收到了您的消息：'{user_input[:50]}...'。我正在分析您的需求，会尽快给您一个合适的回应。如果您有具体问题，请详细说明。"

    def _is_response_adequate(self, response: str, decision: Dict[str, Any]) -> bool:
        """
        判断响应是否充分
        
        Args:
            response: Agent生成的响应
            decision: 决策信息
            
        Returns:
            是否充分
        """
        if not response or len(response.strip()) == 0:
            return False
        
        # 检查是否包含错误提示
        error_indicators = ['错误', 'error', '失败', 'failed', '异常', 'exception']
        if any(indicator in response.lower() for indicator in error_indicators):
            return False
        
        # 检查响应长度（太短可能不充分）
        if len(response) < 10:
            return False
        
        # 如果决策要求工具执行但没有工具调用结果，可能不充分
        action_plan = decision.get('action_plan', {})
        primary_action = action_plan.get('primary_action', '')
        
        if primary_action == 'use_tools':
            tool_calls = action_plan.get('tool_calls', [])
            if tool_calls and '工具调用失败' in response:
                return False
        
        return True

    def _calculate_reward(self, execution_metrics: Dict[str, Any], cognitive_state: Dict[str, Any]) -> float:
        """
        计算奖励信号（用于强化学习）
        
        Args:
            execution_metrics: 执行指标
            cognitive_state: 认知状态
            
        Returns:
            奖励值 (-1.0 到 1.0)
        """
        reward = 0.0
        
        # 1. 基于执行成功与否
        errors = execution_metrics.get('errors', 0)
        if errors == 0:
            reward += 0.5
        else:
            reward -= 0.5 * errors
        
        # 2. 基于执行时间
        exec_time = execution_metrics.get('execution_time', 0)
        if exec_time < 1.0:
            reward += 0.2
        elif exec_time > 5.0:
            reward -= 0.2
        
        # 3. 基于置信度
        confidence = cognitive_state.get('confidence_score', 0.5)
        reward += (confidence - 0.5) * 0.3
        
        # 4. 基于工具使用效率
        tool_calls = execution_metrics.get('tool_calls', 0)
        if tool_calls > 0 and tool_calls <= 3:
            reward += 0.1
        elif tool_calls > 5:
            reward -= 0.1
        
        # 限制在 [-1.0, 1.0] 范围内
        return max(-1.0, min(1.0, reward))

    def _evaluate_response(self, response: str) -> float:
        """
        评估响应质量
        
        Args:
            response: Agent响应文本
            
        Returns:
            质量评分 (0.0-1.0)
        """
        if not response:
            return 0.0
        
        score = 0.5  # 基础分
        
        # 1. 长度适中
        length = len(response)
        if 50 <= length <= 500:
            score += 0.2
        elif length > 1000:
            score -= 0.1
        
        # 2. 不包含错误提示
        if '错误' not in response and 'error' not in response.lower():
            score += 0.2
        
        # 3. 包含有用信息
        if any(keyword in response for keyword in ['建议', '分析', '解释', '帮助']):
            score += 0.1
        
        return max(0.0, min(1.0, score))

    def _perceive_environment(self, user_input: str) -> Dict[str, Any]:
        """环境感知阶段"""
        system_resources = self.environment_monitor.get_metrics_summary()
        user_intent = self.intent_analyzer.analyze(user_input)
        user_emotion = self.emotion_detector.detect(user_input)
        current_context = self.memory_system.get_current_context()

        # 检查资源限制
        resource_constraints = self._assess_resource_constraints(system_resources)

        perception = {
            'user_input': user_input,
            'system_resources': system_resources,
            'user_intent': user_intent,
            'user_emotion': user_emotion,
            'current_context': current_context,
            'resource_constraints': resource_constraints,
            'timestamp': datetime.now().isoformat()
        }

        self.logger.debug(f"感知结果: {perception}")
        return perception

    def _cognitive_processing(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """认知处理阶段 - 元认知评估"""
        # 检索相关记忆
        relevant_memories = self.memory_system.retrieve_relevant_knowledge(
            perception['user_input'],
            context=perception['current_context']
        )

        # 元认知评估
        metacognitive_assessment = self.metacognitive_engine.evaluate_situation(
            user_input=perception['user_input'],
            context=perception['current_context'],
            available_resources=perception['system_resources'],
            relevant_memories=relevant_memories,
            user_intent=perception['user_intent'],
            user_emotion=perception['user_emotion']
        )

        # 策略选择
        strategy = self.strategy_selector.select_strategy(
            metacognitive_assessment=metacognitive_assessment,
            available_tools=self.tool_registry.get_available_tools(),
            resource_constraints=perception['resource_constraints']
        )

        cognitive_state = {
            'metacognitive_assessment': metacognitive_assessment,
            'selected_strategy': strategy,
            'relevant_memories': relevant_memories,
            'confidence_score': metacognitive_assessment.get('overall_confidence', 0.5),
            'risk_level': metacognitive_assessment.get('risk_level', 0.0)
        }

        self.logger.debug(f"认知状态: {cognitive_state}")
        return cognitive_state

    # 在 agent.py 的 _make_decision 方法中
    def _make_decision(self, cognitive_state, perception):
        decision_outcome = self.decision_maker.make_decision(
            cognitive_state=cognitive_state,
            perception=perception
        )
        return decision_outcome.to_dict()

    def _execute_decision(self, decision: Dict[str, Any], perception: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """执行决策阶段"""
        execution_start = time.time()
        execution_metrics = {'tool_calls': 0, 'memory_operations': 0, 'errors': 0}

        try:
            # 从 action_plan 中获取 requires_confirmation
            action_plan = decision.get('action_plan', {})
            requires_confirmation = action_plan.get('requires_confirmation', False)
            
            if requires_confirmation:
                return self._request_confirmation(decision, perception), execution_metrics

            # 执行主要行动
            primary_action = action_plan.get('primary_action', 'direct_response')
            if primary_action == 'direct_response':
                response = self._generate_direct_response(decision, perception)
            elif primary_action == 'use_tools':
                response = self._execute_tools(decision, perception, execution_metrics)
            elif primary_action == 'escalate':
                response = self._handle_escalation(decision, perception)
            else:
                response = self._generate_default_response(perception)

            execution_time = time.time() - execution_start
            execution_metrics['execution_time'] = execution_time

            # 检查是否需要回退到备选方案
            if not self._is_response_adequate(response, decision):
                response = self._execute_fallback(decision, perception, execution_metrics)

            return response, execution_metrics

        except Exception as e:
            execution_metrics['errors'] += 1
            self.logger.error(f"执行决策时出错: {str(e)}")
            return self._handle_execution_error(e, decision, perception), execution_metrics

    def _learn_from_interaction(self, user_input: str, response: str,
                                perception: Dict[str, Any], cognitive_state: Dict[str, Any],
                                decision: Dict[str, Any], execution_metrics: Dict[str, Any]):
        """从交互中学习，优化未来表现"""
        interaction_data = {
            'user_input': user_input,
            'response': response,
            'perception': perception,
            'cognitive_state': cognitive_state,
            'decision': decision,
            'execution_metrics': execution_metrics,
            'outcome': 'success' if execution_metrics['errors'] == 0 else 'failure',
            'timestamp': datetime.now().isoformat()
        }

        # 记录到经验缓冲区
        self.continuous_learner.record_interaction(interaction_data)

        # 更新记忆系统
        self.memory_system.store_interaction(
            user_input=user_input,
            response=response,
            context=perception['current_context'],
            outcome=interaction_data['outcome']
        )

        # 优化工具使用策略
        if 'tool_calls' in execution_metrics and execution_metrics['tool_calls'] > 0:
            self.tool_engine.optimize_tool_usage(interaction_data)

        self.logger.debug("✅ 学习阶段完成")

        # 计算奖励信号
        reward = self._calculate_reward(execution_metrics, cognitive_state)

        # 记录经验
        current_state = {
            'user_input': perception['user_input'],
            'cognitive_state': cognitive_state,
            'context': perception['current_context']
        }

        next_state = {
            'response_quality': self._evaluate_response(response),
            'updated_context': self.memory_system.get_current_context()
        }

        self.experience_replay.add_experience(
            state=current_state,
            action=decision,
            reward=reward,
            next_state=next_state,
            done=True,
            session_id=self.session_id
        )

        # 记录到策略优化器
        strategy_name = decision.get('strategy_type', 'unknown')
        strategy_category = decision.get('primary_action', 'general')

        success = execution_metrics.get('errors', 0) == 0
        reward = self._calculate_reward(execution_metrics, cognitive_state)
        execution_time = execution_metrics.get('execution_time', 0.0)

        self.strategy_optimizer.record_interaction(
            strategy_name=strategy_name,
            strategy_category=strategy_category,
            success=success,
            reward=reward,
            execution_time=execution_time,
            context={
                'user_intent': perception.get('user_intent'),
                'complexity': cognitive_state.get('metacognitive_assessment', {}).get('task_complexity')
            }
        )

    def _update_agent_state(self, perception: Dict[str, Any], cognitive_state: Dict[str, Any],
                            decision: Dict[str, Any], response: str):
        """更新Agent内部状态"""
        # 更新思维过程
        self.thought_process.append({
            'timestamp': datetime.now().isoformat(),
            'perception': perception,
            'cognition': cognitive_state,
            'decision': decision,
            'response': response
        })

        # 保持最近10条思维记录
        if len(self.thought_process) > 10:
            self.thought_process.pop(0)

        # 更新当前目标
        if cognitive_state['metacognitive_assessment'].get('has_goal'):
            self.current_goal = cognitive_state['metacognitive_assessment']['goal_description']

    def get_current_state(self) -> Dict[str, Any]:
        """获取Agent当前状态"""
        return {
            'is_active': self.is_active,
            'current_goal': self.current_goal,
            'thoughts': self.thought_process[-1] if self.thought_process else None,
            'memory_stats': self.memory_system.get_statistics(),
            'performance_metrics': self.performance_tracker.get_summary(),
            'resource_usage': self.environment_monitor.get_metrics_summary(),
            'session_id': self.session_id
        }

    def get_capabilities(self) -> List[str]:
        """获取Agent当前能力列表"""
        capabilities = [
            f"🧠 记忆能力: {self.memory_system.get_capacity_description()}",
            f"🔍 分析能力: 意图识别、情绪检测、环境监控",
            f"🛠️ 工具能力: {', '.join(self.tool_registry.get_tool_names())}",
            f"⚡ 性能: CPU优化，{self.config.CPU_CORES}核心，{self.config.MEMORY_GB}GB内存"
        ]
        return capabilities

    def save_state(self):
        """保存Agent状态到持久化存储"""
        state = {
            'session_id': self.session_id,
            'current_goal': self.current_goal,
            'thought_process': self.thought_process,
            'performance_data': self.performance_tracker.get_all_data(),
            'timestamp': datetime.now().isoformat()
        }

        # 保存记忆系统状态
        self.memory_system.save_to_persistence()

        self.logger.info(f"💾 Agent状态已保存 (会话ID: {self.session_id})")
        return state

    def start_monitoring(self):
        """启动性能监控"""
        self.performance_tracker.start_monitoring()

    def stop_monitoring(self):
        """停止性能监控"""
        self.performance_tracker.stop_monitoring()

    def _resource_monitoring_loop(self):
        """后台资源监控循环"""
        while self.is_active:
            try:
                current_status = self.environment_monitor.get_metrics_summary()

                # 检查资源警报
                if current_status.get('current', {}).get('memory_percent', 0) > self.config.MAX_MEMORY_USAGE * 100:
                    self.logger.warning(f"⚠️ 内存使用率过高: {current_status['current']['memory_percent']:.1f}%")
                    self._optimize_memory_usage()

                if current_status.get('current', {}).get('cpu_percent', 0) > 90:
                    self.logger.warning(f"⚠️ CPU负载过高: {current_status['current']['cpu_percent']:.1f}%")
                    self._optimize_cpu_usage()

                time.sleep(5)  # 每5秒检查一次

            except Exception as e:
                self.logger.error(f"资源监控错误: {str(e)}")
                time.sleep(10)

    def _memory_maintenance_loop(self):
        """后台记忆维护循环"""
        while self.is_active:
            try:
                # 定期进行记忆折叠
                if self.memory_system.needs_folding():
                    self.memory_system.perform_memory_folding()
                    self.logger.info("🧹 记忆折叠完成")

                # 清理过期记忆
                cleaned_count = self.memory_system.clean_expired_memories()
                if cleaned_count > 0:
                    self.logger.info(f"🧹 清理了 {cleaned_count} 条过期记忆")

                time.sleep(300)  # 每5分钟执行一次

            except Exception as e:
                self.logger.error(f"记忆维护错误: {str(e)}")
                time.sleep(60)

    def _continuous_learning_loop(self):
        """后台学习优化循环"""
        while self.is_active:
            try:
                # 检查是否需要批量学习
                if self.experience_replay.needs_batch_learning():
                    analysis = self.experience_replay.perform_batch_analysis()
                    self.logger.info(f"📚 批量分析完成: {analysis}")

                    # 优化策略
                    optimizations = self.experience_replay.optimize_strategies()
                    self.logger.info(f"🎯 策略优化: {optimizations}")

                # 清理过期经验
                self.experience_replay.clear_old_experiences()

                time.sleep(600)  # 每10分钟执行一次
            except Exception as e:
                self.logger.error(f"学习优化错误: {str(e)}")
                time.sleep(300)

    @staticmethod
    def _generate_session_id() -> str:
        """生成唯一的会话ID"""
        import uuid
        return f"session_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _generate_interaction_id() -> str:
        """生成唯一的交互ID"""
        import uuid
        return f"interaction_{uuid.uuid4().hex[:6]}"

    def _assess_resource_constraints(self, system_resources: Dict[str, Any]) -> Dict[str, Any]:
        """评估当前资源约束"""
        # 从嵌套结构中获取资源使用情况
        current = system_resources.get('current', {})
        memory_percent = current.get('memory_percent', 0) / 100.0  # 转换为小数 (0-1)
        cpu_percent = current.get('cpu_percent', 0) / 100.0  # 转换为小数 (0-1)
        
        memory_constraint = memory_percent > self.config.MAX_MEMORY_USAGE
        cpu_constraint = cpu_percent > 0.85

        return {
            'memory_constrained': memory_constraint,
            'cpu_constrained': cpu_constraint,
            'max_response_time': self.config.MAX_RESPONSE_TIME if not (memory_constraint or cpu_constraint) else 2.0,
            'max_context_length': self.config.MAX_CONTEXT_LENGTH if not memory_constraint else 1024
        }

    def _optimize_memory_usage(self):
        """优化内存使用"""
        self.logger.info("🔧 优化内存使用...")

        # 清理缓存
        self.memory_system.clear_caches()

        # 减少工作记忆大小
        self.memory_system.reduce_working_memory_capacity()

        # 释放不重要的记忆
        self.memory_system.release_low_priority_memories()

    def _optimize_cpu_usage(self):
        """优化CPU使用"""
        self.logger.info("🔧 优化CPU使用...")

        # 降低批处理大小
        if hasattr(self.intent_analyzer, 'batch_size'):
            self.intent_analyzer.batch_size = max(1, self.intent_analyzer.batch_size // 2)

        # 减少并行操作
        self.config.THREAD_COUNT = max(1, self.config.THREAD_COUNT // 2)

    def _generate_direct_response(self, decision: Dict[str, Any], perception: Dict[str, Any]) -> str:
        """
        生成直接响应
        
        Args:
            decision: 决策信息
            perception: 感知信息
            
        Returns:
            响应文本
        """
        # 从决策中提取响应
        response = decision.get('response', '')
        
        # 如果决策中没有响应，使用默认响应
        if not response:
            response = self._generate_default_response(perception)
        
        return response

    def _execute_tools(self, decision: Dict[str, Any], perception: Dict[str, Any], 
                       execution_metrics: Dict[str, Any]) -> str:
        """
        执行工具调用
        
        Args:
            decision: 决策信息
            perception: 感知信息
            execution_metrics: 执行指标
            
        Returns:
            工具执行结果
        """
        action_plan = decision.get('action_plan', {})
        tool_calls = action_plan.get('tool_calls', [])
        
        if not tool_calls:
            return self._generate_default_response(perception)
        
        results = []
        for tool_call in tool_calls:
            try:
                tool_name = tool_call.get('tool_name', '')
                tool_params = tool_call.get('params', {})
                
                result = self.tool_engine.execute_tool(tool_name, **tool_params)
                results.append(result)
                execution_metrics['tool_calls'] += 1
            except Exception as e:
                self.logger.error(f"工具调用失败 {tool_name}: {str(e)}")
                execution_metrics['errors'] += 1
        
        # 整合工具结果生成响应
        if results:
            return f"已完成相关操作。结果：{str(results[:3])}"
        return "工具执行完成。"

    def _execute_fallback(self, decision: Dict[str, Any], perception: Dict[str, Any],
                          execution_metrics: Dict[str, Any]) -> str:
        """
        执行备选方案
        
        Args:
            decision: 决策信息
            perception: 感知信息
            execution_metrics: 执行指标
            
        Returns:
            备选方案响应
        """
        action_plan = decision.get('action_plan', {})
        fallback_actions = action_plan.get('fallback_actions', [])
        
        self.logger.info(f"🔄 尝试备选方案: {len(fallback_actions)} 个")
        
        for fallback in fallback_actions:
            if fallback == 'escalate':
                return self._handle_escalation(decision, perception)
            elif fallback == 'clarify':
                return "我需要更多信息来准确回答您的问题。能否请您详细说明一下？"
            elif fallback == 'direct_response':
                return self._generate_default_response(perception)
        
        return "抱歉，当前无法处理您的需求。请稍后再试或换个方式描述。"

    def _request_confirmation(self, decision: Dict[str, Any], perception: Dict[str, Any]) -> str:
        """
        请求用户确认
        
        Args:
            decision: 决策信息
            perception: 感知信息
            
        Returns:
            确认请求文本
        """
        action_plan = decision.get('action_plan', {})
        tool_calls = action_plan.get('tool_calls', [])
        
        if tool_calls:
            tool_names = [t.get('tool_name', '未知工具') for t in tool_calls]
            return f"我需要执行以下操作：{', '.join(tool_names)}。您确认要继续吗？"
        
        return "为了确保准确性，我需要您确认一些信息。请提供更多细节。"

    def _handle_escalation(self, decision: Dict[str, Any], perception: Dict[str, Any]) -> str:
        """
        处理升级（无法处理时的兜底响应）
        
        Args:
            decision: 决策信息
            perception: 感知信息
            
        Returns:
            升级响应文本
        """
        user_input = perception.get('user_input', '')
        return (
            f"抱歉，我目前无法完全处理您的需求：'{user_input[:50]}...'。\n"
            "这可能是由于问题复杂度超出我的能力范围，或者需要更多上下文信息。\n"
            "建议您：\n"
            "1. 简化问题或分步骤描述\n"
            "2. 提供更多背景信息\n"
            "3. 尝试换一种方式表达"
        )

    def __del__(self):
        """析构函数，确保资源清理"""
        try:
            if hasattr(self, 'is_active'):
                self.is_active = False

            if hasattr(self, 'session_id'):
                self.save_state()

            if hasattr(self, 'logger'):
                self.logger.info("👋 Dr.Zero Agent已安全关闭")
        except Exception:
            pass
