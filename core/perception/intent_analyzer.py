"""
意图分析器 - Intent Analyzer
负责理解用户输入的真实意图，识别任务类型、紧急程度和所需能力
为Agent的决策和行动提供语义理解基础
"""
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class IntentType(Enum):
    """意图类型枚举"""
    QUERY = "query"  # 信息查询
    ANALYSIS = "analysis"  # 分析请求
    PROBLEM_SOLVING = "problem_solving"  # 问题解决
    CREATION = "creation"  # 内容创作
    CODE_ASSISTANCE = "code_assistance"  # 代码辅助
    DOCUMENT_PROCESSING = "document_processing"  # 文档处理
    DATA_ANALYSIS = "data_analysis"  # 数据分析
    CLARIFICATION = "clarification"  # 澄清请求
    CONVERSATION = "conversation"  # 闲聊对话
    TASK_EXECUTION = "task_execution"  # 任务执行
    LEARNING = "learning"  # 学习请求
    UNKNOWN = "unknown"  # 未知意图


class UrgencyLevel(Enum):
    """紧急程度"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ComplexityLevel(Enum):
    """复杂度等级"""
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3
    VERY_COMPLEX = 4


@dataclass
class IntentResult:
    """意图分析结果"""
    intent_type: IntentType
    confidence: float
    urgency: UrgencyLevel
    complexity: ComplexityLevel
    domain: str
    keywords: List[str]
    entities: Dict[str, Any]
    requires_clarification: bool
    clarification_questions: List[str]
    suggested_tools: List[str]
    estimated_steps: int
    raw_input: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "intent_type": self.intent_type.value,
            "confidence": self.confidence,
            "urgency": self.urgency.value,
            "complexity": self.complexity.value,
            "domain": self.domain,
            "keywords": self.keywords,
            "entities": self.entities,
            "requires_clarification": self.requires_clarification,
            "clarification_questions": self.clarification_questions,
            "suggested_tools": self.suggested_tools,
            "estimated_steps": self.estimated_steps,
            "timestamp": self.timestamp.isoformat()
        }


class IntentAnalyzer:
    """
    意图分析器

    功能：
    - 识别用户输入的核心意图
    - 评估任务的紧急程度和复杂度
    - 提取关键信息和实体
    - 判断是否需要澄清
    - 推荐合适的工具和方法
    """

    def __init__(self, model_name: str = None, language: str = "zh-CN"):
        """
        初始化意图分析器

        Args:
            model_name: 使用的模型名称（可选，用于高级语义分析）
            language: 主要语言
        """
        self.model_name = model_name
        self.language = language
        self.logger = logging.getLogger("IntentAnalyzer")

        # 意图识别模式库
        self.intent_patterns = self._build_intent_patterns()

        # 领域关键词库
        self.domain_keywords = self._build_domain_keywords()

        # 紧急程度指示词
        self.urgency_indicators = self._build_urgency_indicators()

        # 复杂度评估规则
        self.complexity_rules = self._build_complexity_rules()

        # 工具映射
        self.tool_mapping = self._build_tool_mapping()

        # 统计信息
        self.analysis_count = 0
        self.intent_distribution: Dict[str, int] = {}

        self.logger.info(f"意图分析器初始化完成 - 语言: {language}")

    def analyze(self, user_input: str, context: Dict[str, Any] = None) -> IntentResult:
        """
        分析用户输入的意图

        Args:
            user_input: 用户输入文本
            context: 上下文信息（可选）

        Returns:
            IntentResult: 意图分析结果
        """
        if not user_input or not user_input.strip():
            return self._create_empty_result(user_input)

        # 预处理
        cleaned_input = self._preprocess_input(user_input)

        # 1. 识别意图类型
        intent_type, intent_confidence = self._identify_intent_type(cleaned_input)

        # 2. 评估紧急程度
        urgency = self._assess_urgency(cleaned_input)

        # 3. 评估复杂度
        complexity = self._assess_complexity(cleaned_input, intent_type)

        # 4. 识别领域
        domain = self._identify_domain(cleaned_input)

        # 5. 提取关键词
        keywords = self._extract_keywords(cleaned_input)

        # 6. 提取实体
        entities = self._extract_entities(cleaned_input)

        # 7. 判断是否需要澄清
        needs_clarification, clarification_questions = self._check_clarification_needed(
            cleaned_input, intent_type, context
        )

        # 8. 推荐工具
        suggested_tools = self._suggest_tools(intent_type, domain)

        # 9. 估计步骤数
        estimated_steps = self._estimate_steps(intent_type, complexity)

        # 创建结果
        result = IntentResult(
            intent_type=intent_type,
            confidence=intent_confidence,
            urgency=urgency,
            complexity=complexity,
            domain=domain,
            keywords=keywords,
            entities=entities,
            requires_clarification=needs_clarification,
            clarification_questions=clarification_questions,
            suggested_tools=suggested_tools,
            estimated_steps=estimated_steps,
            raw_input=user_input
        )

        # 更新统计
        self._update_statistics(result)

        self.logger.debug(
            f"意图分析完成 - 类型: {intent_type.value}, "
            f"置信度: {intent_confidence:.2f}, 领域: {domain}"
        )

        return result

    def _preprocess_input(self, text: str) -> str:
        """预处理输入文本"""
        # 去除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        # 标准化标点
        text = text.replace('？', '?').replace('！', '!')
        return text

    def _identify_intent_type(self, text: str) -> Tuple[IntentType, float]:
        """识别意图类型"""
        scores = {}

        # 对每种意图类型计算匹配分数
        for intent_type, patterns in self.intent_patterns.items():
            score = self._calculate_pattern_match(text, patterns)
            scores[intent_type] = score

        # 找到最高分
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        # 如果分数太低，标记为未知
        if best_score < 0.3:
            return IntentType.UNKNOWN, best_score

        return IntentType(best_intent), best_score

    def _calculate_pattern_match(self, text: str, patterns: List[str]) -> float:
        """计算文本与模式的匹配分数"""
        if not patterns:
            return 0.0

        matches = sum(1 for pattern in patterns if pattern.lower() in text.lower())
        return min(matches / len(patterns), 1.0)

    def _assess_urgency(self, text: str) -> UrgencyLevel:
        """评估紧急程度"""
        text_lower = text.lower()

        # 检查紧急指示词
        critical_matches = sum(
            1 for word in self.urgency_indicators['critical']
            if word in text_lower
        )
        high_matches = sum(
            1 for word in self.urgency_indicators['high']
            if word in text_lower
        )

        if critical_matches > 0 or ('!' in text and text.count('!') >= 2):
            return UrgencyLevel.CRITICAL
        elif high_matches > 0:
            return UrgencyLevel.HIGH
        elif any(word in text_lower for word in self.urgency_indicators['low']):
            return UrgencyLevel.LOW
        else:
            return UrgencyLevel.NORMAL

    def _assess_complexity(self, text: str, intent_type: IntentType) -> ComplexityLevel:
        """评估任务复杂度"""
        complexity_score = 0

        # 基于文本长度
        word_count = len(text.split())
        if word_count > 100:
            complexity_score += 2
        elif word_count > 50:
            complexity_score += 1

        # 基于意图类型
        complexity_base = {
            IntentType.CONVERSATION: 0,
            IntentType.QUERY: 1,
            IntentType.CLARIFICATION: 1,
            IntentType.CODE_ASSISTANCE: 2,
            IntentType.ANALYSIS: 2,
            IntentType.PROBLEM_SOLVING: 3,
            IntentType.CREATION: 2,
            IntentType.DATA_ANALYSIS: 3,
            IntentType.TASK_EXECUTION: 2,
            IntentType.LEARNING: 2,
            IntentType.DOCUMENT_PROCESSING: 2,
            IntentType.UNKNOWN: 1,
        }
        complexity_score += complexity_base.get(intent_type, 1)

        # 基于特殊词汇（多步骤、复杂等）
        complex_indicators = ['复杂', '详细', '全面', '深入', '多步骤', '完整']
        if any(indicator in text for indicator in complex_indicators):
            complexity_score += 1

        # 映射到等级
        if complexity_score <= 1:
            return ComplexityLevel.SIMPLE
        elif complexity_score == 2:
            return ComplexityLevel.MODERATE
        elif complexity_score == 3:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.VERY_COMPLEX

    def _identify_domain(self, text: str) -> str:
        """识别任务领域"""
        domain_scores = {}

        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text.lower())
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores, key=domain_scores.get)

        return "general"

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单实现：提取名词性词汇
        # 实际应用中可以使用NLP库如jieba进行中文分词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}

        words = re.findall(r'[\w]+', text)
        keywords = [word for word in words if word not in stop_words and len(word) > 1]

        return keywords[:10]  # 最多返回10个关键词

    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """提取实体信息"""
        entities = {}

        # 提取数字
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            entities['numbers'] = [float(n) for n in numbers]

        # 提取日期（简单模式）
        dates = re.findall(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', text)
        if dates:
            entities['dates'] = dates

        # 提取邮箱
        emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        if emails:
            entities['emails'] = emails

        # 提取URL
        urls = re.findall(r'https?://[\w\./\-_]+', text)
        if urls:
            entities['urls'] = urls

        # 提取代码片段标记
        if '代码' in text:
            code_languages = re.findall(
                r'(\w+)',
                text)
        if code_languages:
            entities['code_languages'] = code_languages

        return entities


    def _check_clarification_needed(
            self,
            text: str,
            intent_type: IntentType,
            context: Dict[str, Any] = None
    ) -> Tuple[bool, List[str]]:
        """判断是否需要澄清"""
        questions = []

        # 检查模糊词汇
        vague_words = ['某个', '一些', '大概', '可能', '也许', '随便', '都行']
        if any(word in text for word in vague_words):
            questions.append("您能具体说明一下吗？")

        # 查询类意图但缺少关键信息
        if intent_type == IntentType.QUERY:
            if len(text.split()) < 5:
                questions.append("您想了解哪方面的信息？")

        # 代码辅助但没有说明语言或问题
        if intent_type == IntentType.CODE_ASSISTANCE:
            if not any(lang in text.lower() for lang in ['python', 'java', 'javascript', 'c++', '代码']):
                questions.append("请问是什么编程语言的代码？")

        # 复杂任务需要明确目标
        if intent_type in [IntentType.PROBLEM_SOLVING, IntentType.ANALYSIS]:
            if '为什么' not in text and '如何' not in text and '怎么' not in text:
                questions.append("您的具体目标是什么？")

        needs_clarification = len(questions) > 0
        return needs_clarification, questions


    def _suggest_tools(self, intent_type: IntentType, domain: str) -> List[str]:
        """推荐工具"""
        return self.tool_mapping.get(intent_type, [])


    def _estimate_steps(self, intent_type: IntentType, complexity: ComplexityLevel) -> int:
        """估计完成任务所需的步骤数"""
        base_steps = {
            IntentType.CONVERSATION: 1,
            IntentType.QUERY: 2,
            IntentType.CLARIFICATION: 1,
            IntentType.CODE_ASSISTANCE: 3,
            IntentType.ANALYSIS: 4,
            IntentType.PROBLEM_SOLVING: 5,
            IntentType.CREATION: 3,
            IntentType.DATA_ANALYSIS: 4,
            IntentType.TASK_EXECUTION: 3,
            IntentType.LEARNING: 3,
            IntentType.DOCUMENT_PROCESSING: 3,
            IntentType.UNKNOWN: 2,
        }

        base = base_steps.get(intent_type, 2)
        multiplier = {
            ComplexityLevel.SIMPLE: 1,
            ComplexityLevel.MODERATE: 1.5,
            ComplexityLevel.COMPLEX: 2,
            ComplexityLevel.VERY_COMPLEX: 3,
        }

        return int(base * multiplier.get(complexity, 1))


    def _create_empty_result(self, raw_input: str) -> IntentResult:
        """创建空结果"""
        return IntentResult(
            intent_type=IntentType.UNKNOWN,
            confidence=0.0,
            urgency=UrgencyLevel.NORMAL,
            complexity=ComplexityLevel.SIMPLE,
            domain="general",
            keywords=[],
            entities={},
            requires_clarification=True,
            clarification_questions=["请输入您的问题"],
            suggested_tools=[],
            estimated_steps=0,
            raw_input=raw_input
        )


    def _update_statistics(self, result: IntentResult):
        """更新统计信息"""
        self.analysis_count += 1
        intent_key = result.intent_type.value
        self.intent_distribution[intent_key] = self.intent_distribution.get(intent_key, 0) + 1


    def get_statistics(self) -> Dict[str, Any]:
        """获取分析统计信息"""
        return {
            "total_analyses": self.analysis_count,
            "intent_distribution": self.intent_distribution.copy(),
            "most_common_intent": max(
                self.intent_distribution,
                key=self.intent_distribution.get
            ) if self.intent_distribution else None
        }


    def _build_intent_patterns(self) -> Dict[str, List[str]]:
        """构建意图模式库"""
        return {
            "query": [
                "什么是", "谁知道", "告诉我", "介绍一下", "查询",
                "what is", "who is", "when", "where", "how to"
            ],
            "analysis": [
                "分析", "评估", "比较", "对比", "优缺点",
                "analyze", "compare", "evaluate", "review"
            ],
            "problem_solving": [
                "解决", "怎么办", "如何处理", "为什么", "怎样",
                "solve", "fix", "troubleshoot", "debug"
            ],
            "creation": [
                "写一个", "创作", "生成", "设计", "制定",
                "write", "create", "generate", "design"
            ],
            "code_assistance": [
                "代码", "编程", "函数", "bug", "错误", "实现",
                "code", "program", "function", "error", "implement"
            ],
            "document_processing": [
                "文档", "文件", "总结", "摘要", "提取",
                "document", "file", "summarize", "extract"
            ],
            "data_analysis": [
                "数据", "统计", "图表", "趋势", "可视化",
                "data", "statistics", "chart", "trend", "visualize"
            ],
            "clarification": [
                "什么意思", "解释一下", "详细说明", "具体",
                "what do you mean", "explain", "clarify"
            ],
            "conversation": [
                "你好", "谢谢", "再见", "聊天", "最近",
                "hello", "hi", "thanks", "bye", "chat"
            ],
            "task_execution": [
                "执行", "运行", "操作", "完成", "帮我",
                "execute", "run", "perform", "do", "help me"
            ],
            "learning": [
                "学习", "教程", "指南", "入门", "教程",
                "learn", "tutorial", "guide", "beginner", "course"
            ]
        }


    def _build_domain_keywords(self) -> Dict[str, List[str]]:
        """构建领域关键词库"""
        return {
            "technology": ["技术", "软件", "硬件", "网络", "系统", "算法"],
            "programming": ["编程", "代码", "开发", "框架", "库", "API"],
            "business": ["商业", "市场", "营销", "销售", "管理", "战略"],
            "education": ["教育", "学习", "课程", "考试", "学校", "知识"],
            "science": ["科学", "研究", "实验", "理论", "物理", "化学"],
            "health": ["健康", "医疗", "运动", "营养", "健身"],
            "finance": ["金融", "投资", "股票", "银行", "理财"],
            "general": ["一般", "普通", "常见", "基本"]
        }


    def _build_urgency_indicators(self) -> Dict[str, List[str]]:
        """构建紧急程度指示词"""
        return {
            "critical": ["紧急", "立刻", "马上", "急", "crucial", "urgent", "asap"],
            "high": ["尽快", "优先", "重要", "important", "priority", "soon"],
            "low": ["有空", "慢慢", "不着急", "whenever", "no rush", "casual"]
        }


    def _build_complexity_rules(self) -> Dict[str, Any]:
        """构建复杂度评估规则"""
        return {
            "simple_indicators": ["简单", "基本", "快速", "简要"],
            "complex_indicators": ["复杂", "详细", "全面", "深入", "完整"],
            "multi_step_keywords": ["首先", "然后", "最后", "步骤", "流程"]
        }


    def _build_tool_mapping(self) -> Dict[IntentType, List[str]]:
        """构建工具映射"""
        return {
            IntentType.QUERY: ["web_search", "knowledge_base"],
            IntentType.ANALYSIS: ["document_analysis", "data_analysis"],
            IntentType.PROBLEM_SOLVING: ["reasoning_engine", "web_search"],
            IntentType.CREATION: ["text_generator", "template_engine"],
            IntentType.CODE_ASSISTANCE: ["code_executor", "code_analyzer"],
            IntentType.DOCUMENT_PROCESSING: ["document_analysis", "text_extractor"],
            IntentType.DATA_ANALYSIS: ["calculator", "data_processor"],
            IntentType.TASK_EXECUTION: ["tool_orchestrator"],
            IntentType.LEARNING: ["knowledge_base", "tutorial_engine"],
    }