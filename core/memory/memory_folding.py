"""
记忆折叠 - Memory Folding
将多个相关的细粒度记忆整合成高层次的抽象记忆单元
模拟人类大脑的记忆压缩和知识提炼过程

核心功能：
- 模式识别：发现相似记忆的共性和规律
- 抽象概括：从具体实例中提取通用概念
- 空间优化：减少冗余，提高存储效率
- 知识提炼：将经验转化为可复用的规则
- 关联强化：建立记忆间的高层次联系
"""
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import hashlib


class FoldingStrategy(Enum):
    """折叠策略"""
    TEMPORAL = "temporal"  # 时间折叠：合并相近时间的记忆
    SEMANTIC = "semantic"  # 语义折叠：合并相似内容的记忆
    EPISODIC = "episodic"  # 情景折叠：合并同一事件的多个片段
    PROCEDURAL = "procedural"  # 程序折叠：合并相似的操作步骤
    CONCEPTUAL = "conceptual"  # 概念折叠：提取共同概念


class AbstractionLevel(Enum):
    """抽象层级"""
    CONCRETE = "concrete"  # 具体实例
    INTERMEDIATE = "intermediate"  # 中等抽象
    ABSTRACT = "abstract"  # 高度抽象
    PRINCIPLE = "principle"  # 原理/规则


@dataclass
class FoldedMemory:
    """折叠后的记忆单元"""
    folded_id: str
    original_memory_ids: List[str]  # 原始记忆ID列表
    folded_content: str  # 折叠后的内容
    abstraction_level: AbstractionLevel
    folding_strategy: FoldingStrategy

    # 时间信息
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    # 统计信息
    compression_ratio: float = 0.0  # 压缩率（原始数量/折叠后数量）
    access_count: int = 0
    usefulness_score: float = 0.5  # 有用性评分

    # 元数据
    key_concepts: List[str] = field(default_factory=list)  # 关键概念
    patterns_identified: List[str] = field(default_factory=list)  # 识别的模式
    confidence: float = 0.5  # 折叠置信度

    # 关联信息
    related_folded_memories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "folded_id": self.folded_id,
            "original_memory_ids": self.original_memory_ids[:20],  # 限制数量
            "folded_content": self.folded_content[:500],
            "abstraction_level": self.abstraction_level.value,
            "folding_strategy": self.folding_strategy.value,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "compression_ratio": round(self.compression_ratio, 3),
            "access_count": self.access_count,
            "usefulness_score": round(self.usefulness_score, 3),
            "key_concepts": self.key_concepts,
            "patterns_identified": self.patterns_identified,
            "confidence": round(self.confidence, 3),
            "related_folded_memories": self.related_folded_memories,
            "tags": self.tags
        }

    def access(self):
        """访问折叠记忆"""
        self.access_count += 1
        self.last_updated = datetime.now()

    def update_usefulness(self, feedback: float):
        """
        更新有用性评分

        Args:
            feedback: 用户反馈 (0.0-1.0)
        """
        # 指数移动平均
        alpha = 0.3
        self.usefulness_score = alpha * feedback + (1 - alpha) * self.usefulness_score


@dataclass
class FoldingCandidate:
    """折叠候选组"""
    candidate_id: str
    memory_ids: List[str]
    similarity_score: float  # 相似度
    folding_strategy: FoldingStrategy
    estimated_compression: float  # 预估压缩率
    key_themes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "memory_count": len(self.memory_ids),
            "similarity_score": round(self.similarity_score, 3),
            "folding_strategy": self.folding_strategy.value,
            "estimated_compression": round(self.estimated_compression, 3),
            "key_themes": self.key_themes
        }


class MemoryFoldingEngine:
    """
    记忆折叠引擎

    功能：
    - 自动检测可折叠的记忆组
    - 执行记忆折叠操作
    - 维护折叠历史的追溯
    - 评估折叠效果和质量
    - 支持折叠回滚

    设计原则：
    - 无损优先：尽量保留原始信息
    - 渐进式：分层次逐步抽象
    - 可追溯：保持与原始记忆的关联
    - 自适应：根据使用反馈调整策略
    """

    def __init__(self,
                 min_similarity_threshold: float = 0.6,
                 min_group_size: int = 3,
                 max_group_size: int = 20,
                 folding_interval_hours: int = 12):
        """
        初始化记忆折叠引擎

        Args:
            min_similarity_threshold: 最小相似度阈值
            min_group_size: 最小折叠组大小
            max_group_size: 最大折叠组大小
            folding_interval_hours: 折叠间隔（小时）
        """
        self.min_similarity_threshold = min_similarity_threshold
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self.folding_interval_hours = folding_interval_hours
        self.logger = logging.getLogger("MemoryFolding")

        # 折叠后的记忆存储
        self.folded_memories: Dict[str, FoldedMemory] = {}

        # 折叠历史（用于追溯）
        self.folding_history: List[Dict[str, Any]] = []

        # 统计信息
        self.total_folding_operations = 0
        self.total_memories_folded = 0
        self.total_space_saved = 0  # 节省的记忆项数量

        # 最后折叠时间
        self.last_folding_time = datetime.now()

        # 折叠策略权重
        self.strategy_weights = {
            FoldingStrategy.TEMPORAL: 0.2,
            FoldingStrategy.SEMANTIC: 0.35,
            FoldingStrategy.EPISODIC: 0.2,
            FoldingStrategy.PROCEDURAL: 0.15,
            FoldingStrategy.CONCEPTUAL: 0.25
        }

        self.logger.info("✅ 记忆折叠引擎初始化完成")

    def detect_folding_candidates(self,
                                  memories: List[Dict[str, Any]]) -> List[FoldingCandidate]:
        """
        检测可折叠的记忆候选组

        Args:
            memories: 记忆列表（来自工作记忆或长期记忆）

        Returns:
            折叠候选组列表
        """
        try:
            candidates = []

            # 1. 按时间窗口分组（时间折叠）
            temporal_groups = self._group_by_temporal_window(memories)
            for group in temporal_groups:
                if len(group) >= self.min_group_size:
                    candidate = self._create_folding_candidate(
                        group, FoldingStrategy.TEMPORAL
                    )
                    if candidate and candidate.similarity_score >= self.min_similarity_threshold:
                        candidates.append(candidate)

            # 2. 按语义相似性分组（语义折叠）
            semantic_groups = self._group_by_semantic_similarity(memories)
            for group in semantic_groups:
                if len(group) >= self.min_group_size:
                    candidate = self._create_folding_candidate(
                        group, FoldingStrategy.SEMANTIC
                    )
                    if candidate and candidate.similarity_score >= self.min_similarity_threshold:
                        candidates.append(candidate)

            # 3. 按主题/标签分组（概念折叠）
            thematic_groups = self._group_by_themes(memories)
            for group in thematic_groups:
                if len(group) >= self.min_group_size:
                    candidate = self._create_folding_candidate(
                        group, FoldingStrategy.CONCEPTUAL
                    )
                    if candidate and candidate.similarity_score >= self.min_similarity_threshold:
                        candidates.append(candidate)

            # 去重和排序
            unique_candidates = self._deduplicate_candidates(candidates)
            unique_candidates.sort(key=lambda c: c.similarity_score, reverse=True)

            self.logger.debug(f"🔍 检测到 {len(unique_candidates)} 个折叠候选组")

            return unique_candidates

        except Exception as e:
            self.logger.error(f"❌ 检测折叠候选失败: {str(e)}")
            return []

    def execute_folding(self,
                       candidate: FoldingCandidate,
                       memory_contents: Dict[str, str]) -> Optional[FoldedMemory]:
        """
        执行记忆折叠

        Args:
            candidate: 折叠候选
            memory_contents: 记忆ID到内容的映射

        Returns:
            折叠后的记忆，失败返回None
        """
        try:
            if len(candidate.memory_ids) < self.min_group_size:
                self.logger.warning(f"⚠️ 候选组太小: {len(candidate.memory_ids)}")
                return None

            # 获取原始内容
            contents = [
                memory_contents.get(mid, "")
                for mid in candidate.memory_ids
                if mid in memory_contents
            ]

            if not contents:
                return None

            # 生成折叠内容
            folded_content = self._generate_folded_content(contents, candidate)

            # 提取关键概念
            key_concepts = self._extract_key_concepts(contents)

            # 识别模式
            patterns = self._identify_patterns(contents)

            # 创建折叠记忆
            folded_id = self._generate_folded_id(candidate)

            # 确定抽象层级
            abstraction_level = self._determine_abstraction_level(
                len(contents), candidate.similarity_score
            )

            folded_memory = FoldedMemory(
                folded_id=folded_id,
                original_memory_ids=candidate.memory_ids,
                folded_content=folded_content,
                abstraction_level=abstraction_level,
                folding_strategy=candidate.folding_strategy,
                compression_ratio=len(contents),
                key_concepts=key_concepts,
                patterns_identified=patterns,
                confidence=candidate.similarity_score,
                tags=self._extract_common_tags(memory_contents, candidate.memory_ids)
            )

            # 存储
            self.folded_memories[folded_id] = folded_memory

            # 记录历史
            self.folding_history.append({
                "timestamp": datetime.now().isoformat(),
                "candidate": candidate.to_dict(),
                "folded_id": folded_id,
                "strategy": candidate.folding_strategy.value
            })

            # 更新统计
            self.total_folding_operations += 1
            self.total_memories_folded += len(candidate.memory_ids)
            self.total_space_saved += len(candidate.memory_ids) - 1

            self.last_folding_time = datetime.now()

            self.logger.info(
                f"✨ 记忆折叠完成: {len(candidate.memory_ids)} -> 1 "
                f"(策略: {candidate.folding_strategy.value})"
            )

            return folded_memory

        except Exception as e:
            self.logger.error(f"❌ 执行记忆折叠失败: {str(e)}")
            return None

    def unfold_memory(self, folded_id: str) -> Optional[List[str]]:
        """
        展开折叠记忆（恢复原始记忆ID列表）

        Args:
            folded_id: 折叠记忆ID

        Returns:
            原始记忆ID列表
        """
        if folded_id not in self.folded_memories:
            return None

        folded_memory = self.folded_memories[folded_id]
        folded_memory.access()

        return folded_memory.original_memory_ids.copy()

    def get_relevant_folded_memories(self,
                                     query: str,
                                     top_k: int = 5) -> List[FoldedMemory]:
        """
        检索相关的折叠记忆

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            相关折叠记忆列表
        """
        scored_memories = []

        for folded_mem in self.folded_memories.values():
            score = self._calculate_folded_relevance(query, folded_mem)
            if score > 0:
                scored_memories.append((score, folded_mem))

        # 排序并返回top_k
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        results = [mem for _, mem in scored_memories[:top_k]]

        # 标记为已访问
        for mem in results:
            mem.access()

        return results

    def evaluate_folding_quality(self, folded_id: str) -> Dict[str, Any]:
        """
        评估折叠质量

        Args:
            folded_id: 折叠记忆ID

        Returns:
            质量评估报告
        """
        if folded_id not in self.folded_memories:
            return {"error": "折叠记忆不存在"}

        folded_mem = self.folded_memories[folded_id]

        # 计算各项指标
        compression_efficiency = folded_mem.compression_ratio / self.max_group_size

        # 使用频率评分
        usage_score = min(folded_mem.access_count / 10.0, 1.0)

        # 置信度评分
        confidence_score = folded_mem.confidence

        # 综合质量得分
        quality_score = (
            compression_efficiency * 0.3 +
            usage_score * 0.3 +
            confidence_score * 0.2 +
            folded_mem.usefulness_score * 0.2
        )

        return {
            "folded_id": folded_id,
            "quality_score": round(quality_score, 3),
            "compression_efficiency": round(compression_efficiency, 3),
            "usage_score": round(usage_score, 3),
            "confidence_score": round(confidence_score, 3),
            "usefulness_score": round(folded_mem.usefulness_score, 3),
            "original_count": len(folded_mem.original_memory_ids),
            "access_count": folded_mem.access_count,
            "abstraction_level": folded_mem.abstraction_level.value,
            "recommendation": self._get_quality_recommendation(quality_score)
        }

    def cleanup_low_quality_foldings(self, quality_threshold: float = 0.3) -> int:
        """
        清理低质量的折叠记忆

        Args:
            quality_threshold: 质量阈值

        Returns:
            清理的数量
        """
        low_quality_ids = []

        for folded_id, folded_mem in self.folded_memories.items():
            quality = self.evaluate_folding_quality(folded_id)
            if quality.get("quality_score", 1.0) < quality_threshold:
                low_quality_ids.append(folded_id)

        for folded_id in low_quality_ids:
            del self.folded_memories[folded_id]

        if low_quality_ids:
            self.logger.info(f"🗑️ 清理了 {len(low_quality_ids)} 个低质量折叠记忆")

        return len(low_quality_ids)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 按策略统计
        strategy_stats = defaultdict(int)
        for mem in self.folded_memories.values():
            strategy_stats[mem.folding_strategy.value] += 1

        # 按抽象层级统计
        abstraction_stats = defaultdict(int)
        for mem in self.folded_memories.values():
            abstraction_stats[mem.abstraction_level.value] += 1

        # 平均压缩率
        avg_compression = (
            sum(m.compression_ratio for m in self.folded_memories.values()) /
            len(self.folded_memories)
            if self.folded_memories else 0
        )

        # 平均质量
        avg_quality = (
            sum(m.usefulness_score for m in self.folded_memories.values()) /
            len(self.folded_memories)
            if self.folded_memories else 0
        )

        return {
            "total_folded_memories": len(self.folded_memories),
            "total_folding_operations": self.total_folding_operations,
            "total_memories_folded": self.total_memories_folded,
            "total_space_saved": self.total_space_saved,
            "average_compression_ratio": round(avg_compression, 3),
            "average_quality_score": round(avg_quality, 3),
            "strategy_distribution": dict(strategy_stats),
            "abstraction_distribution": dict(abstraction_stats),
            "last_folding_time": self.last_folding_time.isoformat()
        }

    def needs_folding(self) -> bool:
        """判断是否需要执行折叠"""
        # 检查时间间隔
        hours_since_last = (
            datetime.now() - self.last_folding_time
        ).total_seconds() / 3600

        if hours_since_last < self.folding_interval_hours:
            return False

        # 如果有足够的折叠记忆，说明有折叠活动
        return len(self.folded_memories) > 0 or True  # 总是允许检查

    def _group_by_temporal_window(self,
                                  memories: List[Dict[str, Any]],
                                  window_hours: int = 2) -> List[List[str]]:
        """按时间窗口分组"""
        if not memories:
            return []

        # 按时间排序
        sorted_memories = sorted(
            memories,
            key=lambda m: m.get("timestamp", m.get("created_at", ""))
        )

        groups = []
        current_group = [sorted_memories[0].get("item_id", sorted_memories[0].get("id"))]
        current_time = self._parse_timestamp(sorted_memories[0])

        for memory in sorted_memories[1:]:
            mem_time = self._parse_timestamp(memory)
            time_diff = (mem_time - current_time).total_seconds() / 3600

            if time_diff <= window_hours:
                current_group.append(memory.get("item_id", memory.get("id")))
            else:
                if len(current_group) >= self.min_group_size:
                    groups.append(current_group)
                current_group = [memory.get("item_id", memory.get("id"))]
                current_time = mem_time

        if len(current_group) >= self.min_group_size:
            groups.append(current_group)

        return groups

    def _group_by_semantic_similarity(self,
                                      memories: List[Dict[str, Any]]) -> List[List[str]]:
        """按语义相似性分组（简化版，实际应使用向量相似度）"""
        if not memories:
            return []

        # 基于关键词重叠的简单相似度
        groups = []
        used_ids = set()

        for i, mem1 in enumerate(memories):
            if mem1.get("item_id", mem1.get("id")) in used_ids:
                continue

            group = [mem1.get("item_id", mem1.get("id"))]
            content1 = mem1.get("content", "").lower()
            words1 = set(content1.split())

            for j, mem2 in enumerate(memories):
                if i == j:
                    continue
                if mem2.get("item_id", mem2.get("id")) in used_ids:
                    continue

                content2 = mem2.get("content", "").lower()
                words2 = set(content2.split())

                # 计算Jaccard相似度
                if words1 and words2:
                    intersection = len(words1 & words2)
                    union = len(words1 | words2)
                    similarity = intersection / union if union > 0 else 0

                    if similarity >= self.min_similarity_threshold:
                        group.append(mem2.get("item_id", mem2.get("id")))

            if len(group) >= self.min_group_size:
                groups.append(group)
                used_ids.update(group)

        return groups

    def _group_by_themes(self, memories: List[Dict[str, Any]]) -> List[List[str]]:
        """按主题/标签分组"""
        if not memories:
            return []

        # 构建标签到记忆ID的映射
        tag_to_ids = defaultdict(list)
        for memory in memories:
            tags = memory.get("tags", [])
            item_id = memory.get("item_id", memory.get("id"))
            for tag in tags:
                tag_to_ids[tag].append(item_id)

        # 收集满足大小的组
        groups = []
        seen_ids = set()

        for tag, ids in tag_to_ids.items():
            unique_ids = [iid for iid in ids if iid not in seen_ids]
            if len(unique_ids) >= self.min_group_size:
                groups.append(unique_ids[:self.max_group_size])
                seen_ids.update(unique_ids)

        return groups

    def _create_folding_candidate(self,
                                  memory_ids: List[str],
                                  strategy: FoldingStrategy) -> Optional[FoldingCandidate]:
        """创建折叠候选"""
        if len(memory_ids) < self.min_group_size:
            return None

        # 估算相似度（简化版）
        similarity = self.min_similarity_threshold + 0.1  # 默认略高于阈值

        # 估算压缩率
        compression = len(memory_ids) / self.max_group_size

        return FoldingCandidate(
            candidate_id=self._generate_candidate_id(memory_ids),
            memory_ids=memory_ids[:self.max_group_size],
            similarity_score=similarity,
            folding_strategy=strategy,
            estimated_compression=compression,
            key_themes=[]  # 可以在这里提取主题
        )

    def _generate_folded_content(self,
                                 contents: List[str],
                                 candidate: FoldingCandidate) -> str:
        """
        生成折叠后的内容

        Args:
            contents: 原始内容列表
            candidate: 折叠候选

        Returns:
            折叠后的摘要内容
        """
        # 简化版：提取共同部分并生成摘要
        # 实际应用中应该调用LLM进行智能摘要

        if not contents:
            return ""

        # 找到最长公共子串（简化）
        common_words = self._find_common_words(contents)

        # 生成摘要
        summary_parts = [
            f"[折叠记忆 - {candidate.folding_strategy.value}]",
            f"包含 {len(contents)} 条相关记忆",
            f"共同主题: {', '.join(common_words[:5])}",
            "",
            "内容摘要:",
        ]

        # 添加代表性片段
        for i, content in enumerate(contents[:3]):
            preview = content[:100] + "..." if len(content) > 100 else content
            summary_parts.append(f"  {i+1}. {preview}")

        if len(contents) > 3:
            summary_parts.append(f"  ... 还有 {len(contents) - 3} 条类似记忆")

        return "\n".join(summary_parts)

    def _extract_key_concepts(self, contents: List[str]) -> List[str]:
        """提取关键概念"""
        # 简化版：提取高频词
        word_freq = defaultdict(int)

        for content in contents:
            words = content.lower().split()
            # 过滤停用词（简化）
            stop_words = {"的", "是", "在", "了", "和", "与", "或", "a", "the", "is", "in"}
            meaningful_words = [w for w in words if w not in stop_words and len(w) > 1]

            for word in meaningful_words:
                word_freq[word] += 1

        # 返回最高频的词
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]

    def _identify_patterns(self, contents: List[str]) -> List[str]:
        """识别模式"""
        patterns = []

        # 检测重复结构
        if len(contents) >= 3:
            patterns.append("重复性模式 detected")

        # 检测时间序列
        patterns.append("时序相关性")

        return patterns

    def _extract_common_tags(self,
                            memory_contents: Dict[str, str],
                            memory_ids: List[str]) -> List[str]:
        """提取共同标签（简化版）"""
        # 实际应该从记忆元数据中提取
        return ["folded", "aggregated"]

    def _determine_abstraction_level(self,
                                     group_size: int,
                                     similarity: float) -> AbstractionLevel:
        """确定抽象层级"""
        if group_size >= 15 and similarity >= 0.8:
            return AbstractionLevel.PRINCIPLE
        elif group_size >= 10 and similarity >= 0.7:
            return AbstractionLevel.ABSTRACT
        elif group_size >= 5:
            return AbstractionLevel.INTERMEDIATE
        else:
            return AbstractionLevel.CONCRETE

    def _calculate_folded_relevance(self,
                                    query: str,
                                    folded_mem: FoldedMemory) -> float:
        """计算折叠记忆的相关性"""
        score = 0.0

        # 内容匹配
        query_words = set(query.lower().split())
        content_words = set(folded_mem.folded_content.lower().split())

        if query_words and content_words:
            overlap = len(query_words & content_words) / len(query_words)
            score += overlap * 0.4

        # 概念匹配
        concept_matches = sum(
            1 for concept in folded_mem.key_concepts
            if concept.lower() in query.lower()
        )
        score += (concept_matches / len(folded_mem.key_concepts)) * 0.3 if folded_mem.key_concepts else 0

        # 有用性加成
        score += folded_mem.usefulness_score * 0.2

        # 访问频率加成
        access_bonus = min(folded_mem.access_count * 0.05, 0.1)
        score += access_bonus

        return min(score, 1.0)

    def _get_quality_recommendation(self, quality_score: float) -> str:
        """获取质量建议"""
        if quality_score >= 0.8:
            return "excellent - 保持"
        elif quality_score >= 0.6:
            return "good - 继续使用"
        elif quality_score >= 0.4:
            return "fair - 观察改进"
        else:
            return "poor - 考虑删除"

    def _deduplicate_candidates(self,
                               candidates: List[FoldingCandidate]) -> List[FoldingCandidate]:
        """去重候选组"""
        if not candidates:
            return []

        unique = []
        seen_memory_sets = []

        for candidate in candidates:
            mem_set = set(candidate.memory_ids)
            is_duplicate = False

            for seen_set in seen_memory_sets:
                # 如果重叠度超过70%，视为重复
                overlap = len(mem_set & seen_set) / max(len(mem_set), len(seen_set))
                if overlap > 0.7:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(candidate)
                seen_memory_sets.append(mem_set)

        return unique

    def _generate_folded_id(self, candidate: FoldingCandidate) -> str:
        """生成折叠记忆ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        hash_input = "".join(sorted(candidate.memory_ids[:5]))
        content_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"folded_{candidate.folding_strategy.value}_{content_hash}_{timestamp}"

    def _generate_candidate_id(self, memory_ids: List[str]) -> str:
        """生成候选ID"""
        hash_input = "".join(sorted(memory_ids[:5]))
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def _find_common_words(self, contents: List[str]) -> List[str]:
        """查找共同词汇"""
        if not contents:
            return []

        word_sets = [set(c.lower().split()) for c in contents]

        # 找到在所有内容中都出现的词
        common = word_sets[0]
        for word_set in word_sets[1:]:
            common = common & word_set

        # 过滤短词
        return [w for w in common if len(w) > 2]

    def _parse_timestamp(self, memory: Dict[str, Any]) -> datetime:
        """解析时间戳"""
        timestamp_str = memory.get("timestamp") or memory.get("created_at")

        if not timestamp_str:
            return datetime.now()

        try:
            if isinstance(timestamp_str, str):
                return datetime.fromisoformat(timestamp_str)
            elif isinstance(timestamp_str, datetime):
                return timestamp_str
        except:
            pass

        return datetime.now()
