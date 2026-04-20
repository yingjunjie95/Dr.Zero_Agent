"""
长期记忆 - Long Term Memory
记忆系统的第三层，负责持久化存储知识和经验
特性：
- 持久化存储：支持磁盘保存和加载
- 向量检索：基于语义相似度的高效检索
- 知识组织：按主题、类型、时间等多维度组织
- 记忆巩固：从工作记忆提取并整合知识
- 智能遗忘：基于重要性和使用频率的自动清理
- 关联网络：维护记忆之间的复杂关系
"""
import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import hashlib
import math


class MemoryCategory(Enum):
    """记忆分类"""
    FACTUAL = "factual"  # 事实性知识
    PROCEDURAL = "procedural"  # 程序性知识（如何做）
    EPISODIC = "episodic"  # 情景记忆（具体事件）
    SEMANTIC = "semantic"  # 语义记忆（概念理解）
    METACOGNITIVE = "metacognitive"  # 元认知（关于认知的知识）
    SOCIAL = "social"  # 社交记忆（用户偏好、交互历史）


class ConsolidationStatus(Enum):
    """巩固状态"""
    RAW = "raw"  # 原始状态，刚从工作记忆转入
    CONSOLIDATING = "consolidating"  # 正在巩固
    CONSOLIDATED = "consolidated"  # 已巩固
    ARCHIVED = "archived"  # 已归档（低优先级）


@dataclass
class LongTermMemoryItem:
    """长期记忆项"""
    item_id: str
    content: str
    category: MemoryCategory
    importance: float  # 重要性 (0.0-1.0)

    # 时间信息
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    last_consolidated: Optional[datetime] = None

    # 访问统计
    access_count: int = 0
    retrieval_count: int = 0  # 被检索到的次数

    # 关联信息
    related_items: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    source_context: Optional[str] = None  # 来源上下文

    # 巩固状态
    consolidation_status: ConsolidationStatus = ConsolidationStatus.RAW
    confidence_score: float = 0.5  # 可信度

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 嵌入向量（用于相似度检索）
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "item_id": self.item_id,
            "content": self.content[:500] if self.content else "",
            "category": self.category.value,
            "importance": round(self.importance, 3),
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "last_consolidated": self.last_consolidated.isoformat() if self.last_consolidated else None,
            "access_count": self.access_count,
            "retrieval_count": self.retrieval_count,
            "related_items": self.related_items[:10],
            "tags": self.tags,
            "source_context": self.source_context,
            "consolidation_status": self.consolidation_status.value,
            "confidence_score": round(self.confidence_score, 3),
            "metadata": self.metadata
        }

    def access(self):
        """访问记忆项"""
        self.last_accessed = datetime.now()
        self.access_count += 1

    def retrieved(self):
        """被检索到"""
        self.retrieval_count += 1
        self.access()

    def calculate_strength(self) -> float:
        """
        计算记忆强度
        基于访问频率、最近访问时间、重要性等因素

        Returns:
            记忆强度 (0.0-1.0)
        """
        now = datetime.now()

        # 1. 基于时间的衰减（距离上次访问的时间）
        time_since_access = (now - self.last_accessed).total_seconds()
        hours_since_access = time_since_access / 3600
        recency_factor = math.exp(-hours_since_access / 168)  # 半衰期约7天

        # 2. 基于访问频率的强化
        frequency_factor = min(math.log(self.access_count + 1, 10), 1.0)

        # 3. 基于检索次数的强化（被检索到说明有价值）
        retrieval_factor = min(math.log(self.retrieval_count + 1, 10), 1.0)

        # 4. 巩固状态加成
        consolidation_bonus = {
            ConsolidationStatus.RAW: 0.0,
            ConsolidationStatus.CONSOLIDATING: 0.1,
            ConsolidationStatus.CONSOLIDATED: 0.2,
            ConsolidationStatus.ARCHIVED: -0.2
        }
        consolidation_factor = consolidation_bonus.get(self.consolidation_status, 0.0)

        # 综合计算
        strength = (
            recency_factor * 0.3 +
            frequency_factor * 0.25 +
            retrieval_factor * 0.25 +
            self.importance * 0.1 +
            consolidation_factor * 0.1
        )

        return max(0.0, min(1.0, strength))

    def is_candidate_for_forgetting(self, threshold: float = 0.15) -> bool:
        """
        判断是否候选遗忘

        Args:
            threshold: 遗忘阈值

        Returns:
            是否应该被遗忘
        """
        strength = self.calculate_strength()
        return strength < threshold and self.access_count < 2


class LongTermMemory:
    """
    长期记忆系统

    功能：
    - 持久化存储：将重要知识保存到磁盘
    - 向量检索：支持语义相似度搜索
    - 记忆巩固：定期处理和强化重要记忆
    - 智能遗忘：自动清理低价值记忆
    - 关联网络：维护记忆间的复杂关系
    - 知识组织：多维度分类和索引

    设计原则：
    - 可扩展：支持大规模记忆存储
    - 高效检索：多种索引加速查询
    - 自适应：根据使用模式动态调整
    - 容错性：数据损坏时能恢复
    """

    def __init__(self,
                 storage_path: str = "data/long_term_memory",
                 max_capacity: int = 10000,
                 forgetting_threshold: float = 0.15,
                 consolidation_interval_hours: int = 24):
        """
        初始化长期记忆

        Args:
            storage_path: 存储路径
            max_capacity: 最大容量（记忆项数量）
            forgetting_threshold: 遗忘阈值
            consolidation_interval_hours: 巩固间隔（小时）
        """
        self.storage_path = storage_path
        self.max_capacity = max_capacity
        self.forgetting_threshold = forgetting_threshold
        self.consolidation_interval_hours = consolidation_interval_hours
        self.logger = logging.getLogger("LongTermMemory")

        # 确保存储目录存在
        os.makedirs(storage_path, exist_ok=True)

        # 记忆存储
        self.memories: Dict[str, LongTermMemoryItem] = {}

        # 索引系统
        self.category_index: Dict[MemoryCategory, Set[str]] = {
            cat: set() for cat in MemoryCategory
        }
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.time_index: List[Tuple[datetime, str]] = []  # (timestamp, item_id)

        # 关联网络
        self.association_graph: Dict[str, Set[str]] = defaultdict(set)

        # 统计信息
        self.total_stored = 0
        self.total_retrieved = 0
        self.total_consolidated = 0
        self.total_forgotten = 0

        # 最后维护时间
        self.last_consolidation = datetime.now()
        self.last_cleanup = datetime.now()

        # 加载已有数据
        self._load_from_disk()

        self.logger.info(f"✅ 长期记忆初始化完成 - 路径: {storage_path}, 容量: {max_capacity}")
        self.logger.info(f"   - 已加载记忆: {len(self.memories)} 条")

    def store(self,
              content: str,
              category: MemoryCategory = MemoryCategory.FACTUAL,
              importance: float = 0.5,
              tags: Optional[List[str]] = None,
              metadata: Optional[Dict[str, Any]] = None,
              source_context: Optional[str] = None,
              related_items: Optional[List[str]] = None,
              embedding: Optional[List[float]] = None) -> str:
        """
        存储记忆到长期记忆

        Args:
            content: 记忆内容
            category: 记忆分类
            importance: 重要性 (0.0-1.0)
            tags: 标签列表
            metadata: 元数据
            source_context: 来源上下文
            related_items: 相关记忆ID列表
            embedding: 嵌入向量

        Returns:
            记忆项ID
        """
        try:
            # 检查容量
            if len(self.memories) >= self.max_capacity:
                self._cleanup_low_strength_memories()

            # 生成ID
            item_id = self._generate_item_id(content)

            # 创建记忆项
            memory_item = LongTermMemoryItem(
                item_id=item_id,
                content=content,
                category=category,
                importance=importance,
                tags=tags or [],
                metadata=metadata or {},
                source_context=source_context,
                related_items=related_items or [],
                embedding=embedding
            )

            # 存储
            self.memories[item_id] = memory_item

            # 更新索引
            self.category_index[category].add(item_id)

            for tag in memory_item.tags:
                self.tag_index[tag].add(item_id)

            self.time_index.append((memory_item.created_at, item_id))
            self.time_index.sort(key=lambda x: x[0], reverse=True)

            # 更新关联网络
            if related_items:
                for related_id in related_items:
                    self.association_graph[item_id].add(related_id)
                    self.association_graph[related_id].add(item_id)

            # 更新统计
            self.total_stored += 1

            self.logger.debug(f"💾 存储长期记忆: ID={item_id[:8]}, 分类={category.value}")

            # 定期保存到磁盘
            if self.total_stored % 10 == 0:
                self._save_to_disk()

            return item_id

        except Exception as e:
            self.logger.error(f"❌ 存储长期记忆失败: {str(e)}")
            return ""

    def retrieve_by_query(self,
                         query: str,
                         top_k: int = 10,
                         categories: Optional[List[MemoryCategory]] = None,
                         min_importance: float = 0.0,
                         tags: Optional[List[str]] = None) -> List[LongTermMemoryItem]:
        """
        通过查询检索相关记忆

        Args:
            query: 查询文本
            top_k: 返回数量
            categories: 分类过滤
            min_importance: 最低重要性
            tags: 标签过滤

        Returns:
            相关记忆项列表（按相关性排序）
        """
        try:
            # 候选集
            candidates = list(self.memories.values())

            # 应用过滤
            if categories:
                candidates = [m for m in candidates if m.category in categories]

            if min_importance > 0:
                candidates = [m for m in candidates if m.importance >= min_importance]

            if tags:
                candidates = [m for m in candidates if any(tag in m.tags for tag in tags)]

            # 计算相关性得分
            scored_candidates = []
            for memory in candidates:
                score = self._calculate_relevance(query, memory)
                if score > 0:
                    scored_candidates.append((score, memory))

            # 排序并返回top_k
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            results = [memory for _, memory in scored_candidates[:top_k]]

            # 标记为已检索
            for memory in results:
                memory.retrieved()
                self.total_retrieved += 1

            self.logger.debug(f"🔍 检索长期记忆: {len(results)} 条结果")

            return results

        except Exception as e:
            self.logger.error(f"❌ 检索长期记忆失败: {str(e)}")
            return []

    def retrieve_by_category(self,
                            category: MemoryCategory,
                            limit: int = 20,
                            sort_by_recency: bool = True) -> List[LongTermMemoryItem]:
        """
        按分类检索记忆

        Args:
            category: 记忆分类
            limit: 最大返回数量
            sort_by_recency: 是否按最近时间排序

        Returns:
            记忆项列表
        """
        item_ids = self.category_index.get(category, set())
        memories = [self.memories[iid] for iid in item_ids if iid in self.memories]

        # 排序
        if sort_by_recency:
            memories.sort(key=lambda m: m.last_accessed, reverse=True)
        else:
            memories.sort(key=lambda m: m.importance, reverse=True)

        return memories[:limit]

    def retrieve_by_tags(self,
                        tags: List[str],
                        match_all: bool = False,
                        limit: int = 20) -> List[LongTermMemoryItem]:
        """
        按标签检索记忆

        Args:
            tags: 标签列表
            match_all: 是否匹配所有标签
            limit: 最大返回数量

        Returns:
            记忆项列表
        """
        if not tags:
            return []

        # 获取候选ID
        candidate_sets = [self.tag_index.get(tag, set()) for tag in tags]

        if match_all:
            matching_ids = set.intersection(*candidate_sets) if candidate_sets else set()
        else:
            matching_ids = set.union(*candidate_sets) if candidate_sets else set()

        # 获取记忆项
        memories = [self.memories[iid] for iid in matching_ids if iid in self.memories]

        # 按重要性排序
        memories.sort(key=lambda m: m.importance, reverse=True)

        return memories[:limit]

    def retrieve_related(self, item_id: str, limit: int = 5) -> List[LongTermMemoryItem]:
        """
        检索相关记忆

        Args:
            item_id: 记忆项ID
            limit: 最大返回数量

        Returns:
            相关记忆项列表
        """
        if item_id not in self.memories:
            return []

        related_ids = self.association_graph.get(item_id, set())
        memories = [self.memories[iid] for iid in related_ids if iid in self.memories]

        # 按强度排序
        memories.sort(key=lambda m: m.calculate_strength(), reverse=True)

        return memories[:limit]

    def consolidate_memory(self, item_id: str) -> bool:
        """
        巩固单个记忆

        Args:
            item_id: 记忆项ID

        Returns:
            是否成功巩固
        """
        if item_id not in self.memories:
            return False

        memory = self.memories[item_id]

        # 更新巩固状态
        memory.consolidation_status = ConsolidationStatus.CONSOLIDATED
        memory.last_consolidated = datetime.now()

        # 提升重要性（巩固的记忆更重要）
        memory.importance = min(1.0, memory.importance + 0.1)

        # 增强置信度
        memory.confidence_score = min(1.0, memory.confidence_score + 0.05)

        self.total_consolidated += 1

        self.logger.info(f"✨ 记忆巩固完成: {item_id[:8]}")
        return True

    def perform_consolidation_cycle(self) -> int:
        """
        执行一轮记忆巩固

        Returns:
            巩固的记忆数量
        """
        now = datetime.now()

        # 检查是否需要巩固
        hours_since_last = (now - self.last_consolidation).total_seconds() / 3600
        if hours_since_last < self.consolidation_interval_hours:
            return 0

        consolidated_count = 0

        # 选择候选记忆进行巩固
        for memory in self.memories.values():
            # 只巩固RAW状态且有一定访问量的记忆
            if (memory.consolidation_status == ConsolidationStatus.RAW and
                memory.access_count >= 2):

                if self.consolidate_memory(memory.item_id):
                    consolidated_count += 1

        self.last_consolidation = now

        if consolidated_count > 0:
            self.logger.info(f"🔄 记忆巩固周期完成: {consolidated_count} 条")
            self._save_to_disk()

        return consolidated_count

    def forget_memories(self) -> int:
        """
        执行遗忘机制，清理低价值记忆

        Returns:
            遗忘的记忆数量
        """
        candidates_for_forgetting = []

        for item_id, memory in self.memories.items():
            if memory.is_candidate_for_forgetting(self.forgetting_threshold):
                candidates_for_forgetting.append(item_id)

        forgotten_count = 0
        for item_id in candidates_for_forgetting:
            if self._remove_memory(item_id):
                forgotten_count += 1
                self.total_forgotten += 1

        if forgotten_count > 0:
            self.logger.info(f"🗑️ 遗忘机制触发: {forgotten_count} 条记忆")
            self._save_to_disk()

        return forgotten_count

    def add_association(self, item_id_1: str, item_id_2: str, strength: float = 1.0):
        """
        添加记忆间的关联

        Args:
            item_id_1: 第一个记忆ID
            item_id_2: 第二个记忆ID
            strength: 关联强度（暂未使用，预留）
        """
        if item_id_1 not in self.memories or item_id_2 not in self.memories:
            self.logger.warning(f"⚠️ 尝试关联不存在的记忆")
            return

        self.association_graph[item_id_1].add(item_id_2)
        self.association_graph[item_id_2].add(item_id_1)

        # 同时更新记忆项中的关联列表
        if item_id_2 not in self.memories[item_id_1].related_items:
            self.memories[item_id_1].related_items.append(item_id_2)

        if item_id_1 not in self.memories[item_id_2].related_items:
            self.memories[item_id_2].related_items.append(item_id_1)

        self.logger.debug(f"🔗 添加关联: {item_id_1[:8]} <-> {item_id_2[:8]}")

    def update_importance(self, item_id: str, new_importance: float):
        """
        更新记忆重要性

        Args:
            item_id: 记忆项ID
            new_importance: 新的重要性值
        """
        if item_id in self.memories:
            old_importance = self.memories[item_id].importance
            self.memories[item_id].importance = max(0.0, min(1.0, new_importance))

            self.logger.debug(
                f"📊 更新重要性: {item_id[:8]} {old_importance:.2f} -> {new_importance:.2f}"
            )

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 按分类统计
        category_stats = {
            cat.value: len(ids) for cat, ids in self.category_index.items()
        }

        # 按巩固状态统计
        consolidation_stats = defaultdict(int)
        for memory in self.memories.values():
            consolidation_stats[memory.consolidation_status.value] += 1

        # 平均强度和重要性
        if self.memories:
            avg_strength = sum(m.calculate_strength() for m in self.memories.values()) / len(self.memories)
            avg_importance = sum(m.importance for m in self.memories.values()) / len(self.memories)
        else:
            avg_strength = 0.0
            avg_importance = 0.0

        return {
            "total_memories": len(self.memories),
            "max_capacity": self.max_capacity,
            "capacity_usage": round(len(self.memories) / self.max_capacity, 3),
            "category_distribution": category_stats,
            "consolidation_distribution": dict(consolidation_stats),
            "average_strength": round(avg_strength, 3),
            "average_importance": round(avg_importance, 3),
            "total_stored": self.total_stored,
            "total_retrieved": self.total_retrieved,
            "total_consolidated": self.total_consolidated,
            "total_forgotten": self.total_forgotten,
            "last_consolidation": self.last_consolidation.isoformat(),
            "last_cleanup": self.last_cleanup.isoformat()
        }

    def save_to_persistence(self):
        """保存到持久化存储"""
        self._save_to_disk()
        self.logger.info("💾 长期记忆已保存到磁盘")

    def clear_all(self):
        """清空所有记忆"""
        self.memories.clear()
        self.category_index = {cat: set() for cat in MemoryCategory}
        self.tag_index.clear()
        self.time_index.clear()
        self.association_graph.clear()

        self.logger.info("🧹 长期记忆已清空")

    def _calculate_relevance(self, query: str, memory: LongTermMemoryItem) -> float:
        """
        计算查询与记忆的相关性

        Args:
            query: 查询文本
            memory: 记忆项

        Returns:
            相关性得分 (0.0-1.0)
        """
        score = 0.0

        # 1. 关键词匹配（简单实现）
        query_words = set(query.lower().split())
        content_words = set(memory.content.lower().split())

        if query_words and content_words:
            overlap = len(query_words & content_words) / len(query_words)
            score += overlap * 0.4

        # 2. 标签匹配
        if memory.tags:
            tag_matches = sum(1 for tag in memory.tags if tag.lower() in query.lower())
            score += (tag_matches / len(memory.tags)) * 0.3 if memory.tags else 0

        # 3. 重要性加成
        score += memory.importance * 0.15

        # 4. 记忆强度加成
        strength = memory.calculate_strength()
        score += strength * 0.15

        return min(score, 1.0)

    def _generate_item_id(self, content: str) -> str:
        """生成记忆项ID"""
        timestamp = datetime.now().isoformat()
        content_hash = hashlib.md5(content[:100].encode()).hexdigest()[:8]
        return f"ltm_{content_hash}_{int(datetime.now().timestamp())}"

    def _remove_memory(self, item_id: str) -> bool:
        """移除单个记忆"""
        if item_id not in self.memories:
            return False

        memory = self.memories[item_id]

        # 从索引中移除
        self.category_index[memory.category].discard(item_id)

        for tag in memory.tags:
            self.tag_index[tag].discard(item_id)

        self.time_index = [(t, iid) for t, iid in self.time_index if iid != item_id]

        # 从关联图中移除
        if item_id in self.association_graph:
            for related_id in self.association_graph[item_id]:
                self.association_graph[related_id].discard(item_id)
            del self.association_graph[item_id]

        # 删除记忆
        del self.memories[item_id]

        return True

    def _cleanup_low_strength_memories(self):
        """清理低强度记忆以释放空间"""
        if len(self.memories) < self.max_capacity * 0.9:
            return

        # 计算所有记忆的强度
        memory_strengths = [
            (mem.calculate_strength(), item_id)
            for item_id, mem in self.memories.items()
        ]

        # 按强度排序
        memory_strengths.sort(key=lambda x: x[0])

        # 移除最弱的20%
        remove_count = int(len(self.memories) * 0.2)
        removed = 0

        for _, item_id in memory_strengths[:remove_count]:
            if self._remove_memory(item_id):
                removed += 1
                self.total_forgotten += 1

        self.logger.info(f"🧹 容量清理: 移除 {removed} 条低强度记忆")
        self.last_cleanup = datetime.now()

    def _save_to_disk(self):
        """保存数据到磁盘"""
        try:
            data = {
                "memories": {
                    item_id: memory.to_dict()
                    for item_id, memory in self.memories.items()
                },
                "association_graph": {
                    k: list(v) for k, v in self.association_graph.items()
                },
                "statistics": {
                    "total_stored": self.total_stored,
                    "total_retrieved": self.total_retrieved,
                    "total_consolidated": self.total_consolidated,
                    "total_forgotten": self.total_forgotten
                },
                "metadata": {
                    "last_saved": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }

            filepath = os.path.join(self.storage_path, "memory_data.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.logger.debug(f"💾 数据已保存到: {filepath}")

        except Exception as e:
            self.logger.error(f"❌ 保存数据失败: {str(e)}")

    def _load_from_disk(self):
        """从磁盘加载数据"""
        try:
            filepath = os.path.join(self.storage_path, "memory_data.json")

            if not os.path.exists(filepath):
                self.logger.info("ℹ️ 未找到现有数据文件，从空状态开始")
                return

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 加载记忆
            for item_id, mem_data in data.get("memories", {}).items():
                memory = LongTermMemoryItem(
                    item_id=mem_data["item_id"],
                    content=mem_data["content"],
                    category=MemoryCategory(mem_data["category"]),
                    importance=mem_data["importance"],
                    created_at=datetime.fromisoformat(mem_data["created_at"]),
                    last_accessed=datetime.fromisoformat(mem_data["last_accessed"]),
                    last_consolidated=(
                        datetime.fromisoformat(mem_data["last_consolidated"])
                        if mem_data.get("last_consolidated") else None
                    ),
                    access_count=mem_data.get("access_count", 0),
                    retrieval_count=mem_data.get("retrieval_count", 0),
                    related_items=mem_data.get("related_items", []),
                    tags=mem_data.get("tags", []),
                    source_context=mem_data.get("source_context"),
                    consolidation_status=ConsolidationStatus(
                        mem_data.get("consolidation_status", "raw")
                    ),
                    confidence_score=mem_data.get("confidence_score", 0.5),
                    metadata=mem_data.get("metadata", {})
                )

                self.memories[item_id] = memory

                # 重建索引
                self.category_index[memory.category].add(item_id)
                for tag in memory.tags:
                    self.tag_index[tag].add(item_id)
                self.time_index.append((memory.created_at, item_id))

            # 重建关联图
            for k, v in data.get("association_graph", {}).items():
                self.association_graph[k] = set(v)

            # 恢复统计
            stats = data.get("statistics", {})
            self.total_stored = stats.get("total_stored", 0)
            self.total_retrieved = stats.get("total_retrieved", 0)
            self.total_consolidated = stats.get("total_consolidated", 0)
            self.total_forgotten = stats.get("total_forgotten", 0)

            self.time_index.sort(key=lambda x: x[0], reverse=True)

            self.logger.info(f"✅ 已加载 {len(self.memories)} 条长期记忆")

        except Exception as e:
            self.logger.error(f"❌ 加载数据失败: {str(e)}")
            self.logger.warning("⚠️ 从空状态开始")
