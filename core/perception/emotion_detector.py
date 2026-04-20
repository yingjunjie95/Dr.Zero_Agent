"""
情绪检测器 - Emotion Detector
负责识别用户输入中的情绪状态和情感倾向
为Agent的响应策略提供情感智能支持
"""
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict


class EmotionType(Enum):
    """情绪类型枚举"""
    JOY = "joy"  # 高兴、满意
    SADNESS = "sadness"  # 悲伤、失落
    ANGER = "anger"  # 愤怒、不满
    FEAR = "fear"  # 恐惧、焦虑
    SURPRISE = "surprise"  # 惊讶、意外
    DISGUST = "disgust"  # 厌恶、反感
    NEUTRAL = "neutral"  # 中性、平静
    CONFUSION = "confusion"  # 困惑、疑惑
    EXCITEMENT = "excitement"  # 兴奋、期待
    FRUSTRATION = "frustration"  # 挫败、沮丧


class SentimentPolarity(Enum):
    """情感极性"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class EmotionResult:
    """情绪检测结果"""
    primary_emotion: EmotionType
    emotion_intensity: float  # 0.0 - 1.0
    secondary_emotions: Dict[EmotionType, float]
    sentiment_polarity: SentimentPolarity
    sentiment_score: float  # -1.0 (负面) to 1.0 (正面)
    emotional_keywords: List[str]
    requires_empathy: bool
    suggested_tone: str
    raw_input: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "primary_emotion": self.primary_emotion.value,
            "emotion_intensity": round(self.emotion_intensity, 3),
            "secondary_emotions": {
                k.value: round(v, 3) for k, v in self.secondary_emotions.items()
            },
            "sentiment_polarity": self.sentiment_polarity.value,
            "sentiment_score": round(self.sentiment_score, 3),
            "emotional_keywords": self.emotional_keywords,
            "requires_empathy": self.requires_empathy,
            "suggested_tone": self.suggested_tone,
            "timestamp": self.timestamp.isoformat()
        }


class EmotionDetector:
    """
    情绪检测器

    功能：
    - 识别用户文本中的主要情绪
    - 评估情绪强度
    - 分析情感极性（正面/负面/中性）
    - 提取情绪关键词
    - 建议Agent的响应语气和策略
    """

    def __init__(self, language: str = "zh-CN"):
        """
        初始化情绪检测器

        Args:
            language: 主要语言
        """
        self.language = language
        self.logger = logging.getLogger("EmotionDetector")

        # 情绪词库
        self.emotion_lexicon = self._build_emotion_lexicon()

        # 强度修饰词
        self.intensity_modifiers = self._build_intensity_modifiers()

        # 否定词
        self.negation_words = self._build_negation_words()

        # 标点符号情绪权重
        self.punctuation_weights = self._build_punctuation_weights()

        # 语气建议映射
        self.tone_suggestions = self._build_tone_suggestions()

        # 统计信息
        self.detection_count = 0
        self.emotion_distribution: Dict[str, int] = defaultdict(int)

        self.logger.info(f"情绪检测器初始化完成 - 语言: {language}")

    def detect(self, text: str, context: Dict[str, Any] = None) -> EmotionResult:
        """
        检测文本中的情绪

        Args:
            text: 用户输入文本
            context: 上下文信息（可选）

        Returns:
            EmotionResult: 情绪检测结果
        """
        if not text or not text.strip():
            return self._create_neutral_result(text)

        # 预处理
        cleaned_text = self._preprocess_text(text)

        # 1. 提取情绪关键词
        emotional_keywords = self._extract_emotional_keywords(cleaned_text)

        # 2. 计算各情绪类型的得分
        emotion_scores = self._calculate_emotion_scores(cleaned_text, emotional_keywords)

        # 3. 应用强度修饰词
        emotion_scores = self._apply_intensity_modifiers(cleaned_text, emotion_scores)

        # 4. 处理否定词
        emotion_scores = self._handle_negations(cleaned_text, emotion_scores)

        # 5. 考虑标点符号的影响
        emotion_scores = self._apply_punctuation_weights(text, emotion_scores)

        # 6. 确定主要情绪
        primary_emotion, intensity = self._determine_primary_emotion(emotion_scores)

        # 7. 获取次要情绪
        secondary_emotions = self._get_secondary_emotions(emotion_scores, primary_emotion)

        # 8. 计算情感极性
        sentiment_polarity, sentiment_score = self._calculate_sentiment(
            emotion_scores, emotional_keywords
        )

        # 9. 判断是否需要共情
        requires_empathy = self._check_empathy_needed(primary_emotion, intensity)

        # 10. 建议响应语气
        suggested_tone = self._suggest_response_tone(primary_emotion, sentiment_polarity)

        # 创建结果
        result = EmotionResult(
            primary_emotion=primary_emotion,
            emotion_intensity=intensity,
            secondary_emotions=secondary_emotions,
            sentiment_polarity=sentiment_polarity,
            sentiment_score=sentiment_score,
            emotional_keywords=emotional_keywords,
            requires_empathy=requires_empathy,
            suggested_tone=suggested_tone,
            raw_input=text
        )

        # 更新统计
        self._update_statistics(result)

        self.logger.debug(
            f"情绪检测完成 - 主要情绪: {primary_emotion.value}, "
            f"强度: {intensity:.2f}, 极性: {sentiment_polarity.value}"
        )

        return result

    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 去除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        # 标准化标点
        text = text.replace('！', '!').replace('？', '?')
        return text.lower() if self.language == "en-US" else text

    def _extract_emotional_keywords(self, text: str) -> List[str]:
        """提取情绪关键词"""
        keywords = []

        for emotion_type, words in self.emotion_lexicon.items():
            for word in words:
                if word in text:
                    keywords.append(word)

        return list(set(keywords))

    def _calculate_emotion_scores(self, text: str, keywords: List[str]) -> Dict[EmotionType, float]:
        """计算各情绪类型的得分"""
        scores = {emotion: 0.0 for emotion in EmotionType}

        # 基于关键词匹配
        for keyword in keywords:
            for emotion_type, words in self.emotion_lexicon.items():
                if keyword in words:
                    scores[emotion_type] += 1.0

        # 归一化分数到 0-1 范围
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {k: min(v / max_score, 1.0) for k, v in scores.items()}

        return scores

    def _apply_intensity_modifiers(self, text: str, scores: Dict[EmotionType, float]) -> Dict[EmotionType, float]:
        """应用强度修饰词"""
        modified_scores = scores.copy()

        # 检查强度修饰词
        high_intensity = any(word in text for word in self.intensity_modifiers['high'])
        low_intensity = any(word in text for word in self.intensity_modifiers['low'])

        if high_intensity:
            # 增强所有非零情绪的强度
            modified_scores = {
                k: min(v * 1.3, 1.0) if v > 0 else v
                for k, v in modified_scores.items()
            }
        elif low_intensity:
            # 减弱情绪强度
            modified_scores = {
                k: v * 0.7 for k, v in modified_scores.items()
            }

        return modified_scores

    def _handle_negations(self, text: str, scores: Dict[EmotionType, float]) -> Dict[EmotionType, float]:
        """处理否定词"""
        modified_scores = scores.copy()

        # 简单实现：如果包含否定词，降低负面情绪强度
        has_negation = any(word in text for word in self.negation_words)

        if has_negation:
            # 降低负面情绪
            negative_emotions = [EmotionType.ANGER, EmotionType.SADNESS, EmotionType.FEAR, EmotionType.DISGUST]
            for emotion in negative_emotions:
                modified_scores[emotion] *= 0.6

        return modified_scores

    def _apply_punctuation_weights(self, original_text: str, scores: Dict[EmotionType, float]) -> Dict[EmotionType, float]:
        """应用标点符号权重"""
        modified_scores = scores.copy()

        # 感叹号增强情绪
        exclamation_count = original_text.count('!') + original_text.count('！')
        if exclamation_count > 0:
            boost = min(exclamation_count * 0.1, 0.3)
            modified_scores = {
                k: min(v + boost, 1.0) if v > 0 else v
                for k, v in modified_scores.items()
            }

        # 问号可能表示困惑
        question_count = original_text.count('?') + original_text.count('？')
        if question_count > 2:
            modified_scores[EmotionType.CONFUSION] = min(
                modified_scores[EmotionType.CONFUSION] + 0.2, 1.0
            )

        return modified_scores

    def _determine_primary_emotion(self, scores: Dict[EmotionType, float]) -> Tuple[EmotionType, float]:
        """确定主要情绪及其强度"""
        if not scores or all(v == 0.0 for v in scores.values()):
            return EmotionType.NEUTRAL, 0.0

        primary = max(scores, key=scores.get)
        intensity = scores[primary]

        # 如果最高分太低，认为是中性
        if intensity < 0.2:
            return EmotionType.NEUTRAL, 0.0

        return primary, intensity

    def _get_secondary_emotions(
        self,
        scores: Dict[EmotionType, float],
        primary: EmotionType
    ) -> Dict[EmotionType, float]:
        """获取次要情绪（得分>0.3且不是主要情绪的）"""
        secondary = {}

        for emotion, score in scores.items():
            if emotion != primary and score >= 0.3:
                secondary[emotion] = score

        # 按得分排序，最多返回3个
        sorted_secondary = dict(
            sorted(secondary.items(), key=lambda x: x[1], reverse=True)[:3]
        )

        return sorted_secondary

    def _calculate_sentiment(
        self,
        scores: Dict[EmotionType, float],
        keywords: List[str]
    ) -> Tuple[SentimentPolarity, float]:
        """计算情感极性"""
        positive_emotions = [EmotionType.JOY, EmotionType.EXCITEMENT, EmotionType.SURPRISE]
        negative_emotions = [EmotionType.SADNESS, EmotionType.ANGER, EmotionType.FEAR,
                           EmotionType.DISGUST, EmotionType.FRUSTRATION]

        positive_score = sum(scores.get(e, 0.0) for e in positive_emotions)
        negative_score = sum(scores.get(e, 0.0) for e in negative_emotions)

        # 计算综合得分 (-1.0 到 1.0)
        total = positive_score + negative_score
        if total == 0:
            return SentimentPolarity.NEUTRAL, 0.0

        sentiment_score = (positive_score - negative_score) / total

        # 确定极性
        if sentiment_score > 0.2:
            polarity = SentimentPolarity.POSITIVE
        elif sentiment_score < -0.2:
            polarity = SentimentPolarity.NEGATIVE
        else:
            polarity = SentimentPolarity.NEUTRAL

        return polarity, sentiment_score

    def _check_empathy_needed(self, emotion: EmotionType, intensity: float) -> bool:
        """判断是否需要共情回应"""
        empathy_emotions = [
            EmotionType.SADNESS,
            EmotionType.ANGER,
            EmotionType.FEAR,
            EmotionType.FRUSTRATION
        ]

        return emotion in empathy_emotions and intensity > 0.4

    def _suggest_response_tone(
        self,
        emotion: EmotionType,
        polarity: SentimentPolarity
    ) -> str:
        """建议响应语气"""
        return self.tone_suggestions.get(emotion, "professional")

    def _create_neutral_result(self, text: str) -> EmotionResult:
        """创建中性结果"""
        return EmotionResult(
            primary_emotion=EmotionType.NEUTRAL,
            emotion_intensity=0.0,
            secondary_emotions={},
            sentiment_polarity=SentimentPolarity.NEUTRAL,
            sentiment_score=0.0,
            emotional_keywords=[],
            requires_empathy=False,
            suggested_tone="professional",
            raw_input=text
        )

    def _update_statistics(self, result: EmotionResult):
        """更新统计信息"""
        self.detection_count += 1
        self.emotion_distribution[result.primary_emotion.value] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """获取检测统计信息"""
        return {
            "total_detections": self.detection_count,
            "emotion_distribution": dict(self.emotion_distribution),
            "most_common_emotion": max(
                self.emotion_distribution,
                key=self.emotion_distribution.get
            ) if self.emotion_distribution else None
        }

    def _build_emotion_lexicon(self) -> Dict[str, List[str]]:
        """构建情绪词库（中英文混合）"""
        return {
            EmotionType.JOY: [
                "开心", "快乐", "高兴", "愉快", "满意", "幸福", "喜悦", "兴奋",
                "happy", "joy", "glad", "pleased", "delighted", "cheerful"
            ],
            EmotionType.SADNESS: [
                "伤心", "难过", "悲伤", "失落", "沮丧", "忧郁", "痛苦", "失望",
                "sad", "unhappy", "depressed", "disappointed", "sorrow", "grief"
            ],
            EmotionType.ANGER: [
                "生气", "愤怒", "恼火", "烦躁", "不满", "气愤", "怒火", "愤慨",
                "angry", "mad", "furious", "annoyed", "irritated", "outraged"
            ],
            EmotionType.FEAR: [
                "害怕", "恐惧", "担心", "焦虑", "紧张", "惊慌", "担忧", "畏惧",
                "afraid", "scared", "fear", "anxious", "worried", "nervous", "panic"
            ],
            EmotionType.SURPRISE: [
                "惊讶", "吃惊", "意外", "震惊", "惊奇", "诧异", "没想到",
                "surprised", "amazed", "shocked", "astonished", "unexpected"
            ],
            EmotionType.DISGUST: [
                "讨厌", "厌恶", "恶心", "反感", "嫌弃", "憎恶",
                "disgusted", "hate", "repulsed", "revolted", "detest"
            ],
            EmotionType.CONFUSION: [
                "困惑", "疑惑", "不明白", "不清楚", "迷茫", "糊涂", "不解",
                "confused", "puzzled", "unclear", "lost", "bewildered"
            ],
            EmotionType.EXCITEMENT: [
                "激动", "兴奋", "期待", "迫不及待", "振奋", "热情",
                "excited", "thrilled", "eager", "enthusiastic", "pumped"
            ],
            EmotionType.FRUSTRATION: [
                "挫败", "无奈", "无力", "受挫", "郁闷", "憋屈",
                "frustrated", "defeated", "helpless", "discouraged"
            ],
            EmotionType.NEUTRAL: [
                "平静", "冷静", "淡定", "一般", "普通",
                "calm", "neutral", "okay", "fine", "normal"
            ]
        }

    def _build_intensity_modifiers(self) -> Dict[str, List[str]]:
        """构建强度修饰词"""
        return {
            "high": [
                "非常", "极其", "特别", "太", "超级", "十分", "格外", "异常",
                "very", "extremely", "really", "so", "incredibly", "absolutely"
            ],
            "low": [
                "有点", "稍微", "略微", "一些", "轻微",
                "a bit", "slightly", "somewhat", "a little", "kind of"
            ]
        }

    def _build_negation_words(self) -> List[str]:
        """构建否定词列表"""
        return [
            "不", "没", "没有", "别", "勿", "非", "未", "莫",
            "not", "no", "never", "neither", "nor", "don't", "doesn't", "didn't"
        ]

    def _build_punctuation_weights(self) -> Dict[str, float]:
        """构建标点符号权重"""
        return {
            "!": 0.1,
            "！": 0.1,
            "?": 0.05,
            "？": 0.05,
            "...": 0.15,
            "……": 0.15
        }

    def _build_tone_suggestions(self) -> Dict[EmotionType, str]:
        """构建语气建议映射"""
        return {
            EmotionType.JOY: "warm_and_friendly",
            EmotionType.SADNESS: "empathetic_and_supportive",
            EmotionType.ANGER: "calm_and_professional",
            EmotionType.FEAR: "reassuring_and_helpful",
            EmotionType.SURPRISE: "engaging_and_informative",
            EmotionType.DISGUST: "respectful_and_solution_focused",
            EmotionType.NEUTRAL: "professional",
            EmotionType.CONFUSION: "patient_and_clear",
            EmotionType.EXCITEMENT: "enthusiastic_and_encouraging",
            EmotionType.FRUSTRATION: "understanding_and_practical"
        }
