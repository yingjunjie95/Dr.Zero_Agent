"""
工作记忆 - Working Memory
记忆系统的第二层，负责当前任务的活跃信息处理
特性：
- 有限容量（类似人类工作记忆7±2原则）
- 快速访问和更新
- 注意力机制
- 与感官记忆和长期记忆的双向交互
- 支持思维链和推理过程
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import hashlib


class MemoryType(Enum):
    """记忆类型"""
    EPISODIC = "episodic"  # 情景记忆（具体事件）
    SEMANTIC = "semantic"  # 语义记忆（事实知识）
    PROCEDURAL = "procedural"  # 程序记忆（技能方法）
    WORKING = "working"  # 工作记忆（当前任务）


class AttentionLevel(Enum):
    """注意力等级"""
    FOCUSED = "focused"  # 高度关注
    ACTIVE = "active"  # 活跃
    MAINTAINED = "maintained"  # 保持
    PERIPHERAL = "peripheral"  # 边缘


@dataclass
class WorkingMemoryItem:
    """工作记忆项"""
    item_id: str
    content: Any
    memory_type: MemoryType
    attention: AttentionLevel
    relevance_score: float  # 相关性得分 (0.0-1.0)

    # 时间信息
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    # 关联信息
    related_items: List[str] = field(default_factory=list)  # 相关项目ID
    source_memory_id: Optional[str] = None  # 来源记忆ID（来自感官或长期记忆）

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # 状态
    is_locked: bool = False  # 是否被锁定（防止被淘汰）
    decay_counter: float = 0.0  # 衰减计数器

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "item_id": self.item_id,
            "content": str(self.content)[:300] if self.content else "",
            "memory_type": self.memory_type.value,
            "attention": self.attention.value,
            "relevance_score": round(self.relevance_score, 3),
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "related_items": self.related_items[:5],  # 限制数量
            "source_memory_id": self.source_memory_id,
            "metadata": self.metadata,
            "tags": self.tags,
            "is_locked": self.is_locked
        }

    def access(self):
        """访问项目"""
        self.last_accessed = datetime.now()
        self.access_count += 1
        # 重置部分衰减
        self.decay_counter = max(0, self.decay_counter - 0.3)

    def increase_attention(self):
        """提升注意力等级"""
        attention_order = [
            AttentionLevel.PERIPHERAL,
            AttentionLevel.MAINTAINED,
            AttentionLevel.ACTIVE,
            AttentionLevel.FOCUSED
        ]
        current_idx = attention_order.index(self.attention)
        if current_idx < len(attention_order) - 1:
            self.attention = attention_order[current_idx + 1]

    def decrease_attention(self):
        """降低注意力等级"""
        attention_order = [
            AttentionLevel.PERIPHERAL,
            AttentionLevel.MAINTAINED,
            AttentionLevel.ACTIVE,
            AttentionLevel.FOCUSED
        ]
        current_idx = attention_order.index(self.attention)
        if current_idx > 0:
            self.attention = attention_order[current_idx - 1]


@dataclass
class ThoughtChain:
    """思维链 - 记录推理过程"""
    chain_id: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    conclusion: Optional[str] = None

    def add_step(self, step_type: str, content: str, confidence: float = 0.5):
        """添加思维步骤"""
        self.steps.append({
            "step_number": len(self.steps) + 1,
            "type": step_type,
            "content": content,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })

    def complete(self, conclusion: str):
        """完成思维链"""
        self.end_time = datetime.now()
        self.conclusion = conclusion

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "steps_count": len(self.steps),
            "steps": self.steps,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "conclusion": self.conclusion
        }


class WorkingMemory:
    """
    工作记忆系统

    功能：
    - 容量管理：维持有限的工作记忆容量
    - 注意力机制：动态调整信息的注意力等级
    - 关联网络：维护记忆项之间的关联
    - 思维链：记录和追踪推理过程
    - 双向同步：与感官记忆和长期记忆交互

    设计原则：
    - 高效访问：O(1)或O(log n)的访问复杂度
    - 智能淘汰：基于注意力、相关性和使用频率
    - 上下文感知：维护当前任务的上下文信息
    """

    def __init__(self, capacity: int = 50, embedding_dim: int = 768):
        """
        初始化工作记忆

        Args:
            capacity: 工作记忆容量
            embedding_dim: 嵌入维度（用于相似度计算）
        """
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.logger = logging.getLogger("WorkingMemory")

        # 记忆存储
        self.items: Dict[str, WorkingMemoryItem] = {}
        self.insertion_order: deque = deque(maxlen=capacity)  # 插入顺序

        # 注意力索引
        self.attention_index: Dict[AttentionLevel, List[str]] = {
            level: [] for level in AttentionLevel
        }

        # 类型索引
        self.type_index: Dict[MemoryType, List[str]] = {
            mtype: [] for mtype in MemoryType
        }

        # 标签索引
        self.tag_index: Dict[str, List[str]] = {}

        # 思维链
        self.active_chains: Dict[str, ThoughtChain] = {}
        self.completed_chains: deque = deque(maxlen=100)

        # 当前上下文
        self.current_context: Dict[str, Any] = {
            "task_description": "",
            "current_focus": [],
            "recent_topics": deque(maxlen=10),
            "user_intent": None,
            "conversation_stage": "initial",
            "active_goals": []
        }

        # 统计信息
        self.total_added = 0
        self.total_removed = 0
        self.total_accessed = 0
        self.total_promoted_to_longterm = 0

        # 最后清理时间
        self.last_cleanup = datetime.now()

        self.logger.info(f"✅ 工作记忆初始化完成 - 容量: {capacity}")

    def add_item(self,
                content: Any,
                memory_type: MemoryType = MemoryType.WORKING,
                attention: AttentionLevel = AttentionLevel.ACTIVE,
                relevance_score: float = 0.5,
                tags: Optional[List[str]] = None,
                metadata: Optional[Dict[str, Any]] = None,
                source_memory_id: Optional[str] = None,
                related_items: Optional[List[str]] = None) -> str:
        """
        添加项目到工作记忆

        Args:
            content: 内容
            memory_type: 记忆类型
            attention: 注意力等级
            relevance_score: 相关性得分
            tags: 标签列表
            metadata: 元数据
            source_memory_id: 来源记忆ID
            related_items: 相关项目ID列表

        Returns:
            项目ID
        """
        # 检查容量，必要时淘汰
        if len(self.items) >= self.capacity:
            self._evict_least_important()

        # 生成ID
        item_id = self._generate_item_id(content)

        # 创建项目
        item = WorkingMemoryItem(
            item_id=item_id,
            content=content,
            memory_type=memory_type,
            attention=attention,
            relevance_score=relevance_score,
            tags=tags or [],
            metadata=metadata or {},
            source_memory_id=source_memory_id,
            related_items=related_items or []
        )

        # 存储
        self.items[item_id] = item
        self.insertion_order.append(item_id)

        # 更新索引
        self.attention_index[attention].append(item_id)
        self.type_index[memory_type].append(item_id)

        for tag in item.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = []
            self.tag_index[tag].append(item_id)

        # 更新统计
        self.total_added += 1

        self.logger.debug(f"💾 添加到工作记忆: ID={item_id[:8]}, 类型={memory_type.value}")

        return item_id

    def retrieve_by_relevance(self,
                             query: str,
                             top_k: int = 10,
                             min_relevance: float = 0.3) -> List[WorkingMemoryItem]:
        """
        按相关性检索记忆

        Args:
            query: 查询内容
            top_k: 返回数量
            min_relevance: 最低相关性阈值

        Returns:
            相关工作记忆项列表
        """
        # 简单的相关性评分（实际应用中应使用向量相似度）
        scored_items = []

        for item in self.items.values():
            # 基础相关性
            score = item.relevance_score

            # 注意力加成
            attention_bonus = {
                AttentionLevel.FOCUSED: 0.3,
                AttentionLevel.ACTIVE: 0.2,
                AttentionLevel.MAINTAINED: 0.1,
                AttentionLevel.PERIPHERAL: 0.0
            }
            score += attention_bonus.get(item.attention, 0.0)

            # 访问频率加成
            frequency_bonus = min(item.access_count * 0.02, 0.2)
            score += frequency_bonus

            # 内容匹配（简单关键词匹配）
            if isinstance(item.content, str) and query:
                query_words = set(query.lower().split())
                content_words = set(item.content.lower().split())
                overlap = len(query_words & content_words) / max(len(query_words), 1)
                score += overlap * 0.3

            if score >= min_relevance:
                scored_items.append((score, item))

        # 排序并返回top_k
        scored_items.sort(key=lambda x: x[0], reverse=True)
        result_items = [item for _, item in scored_items[:top_k]]

        # 标记为已访问
        for item in result_items:
            item.access()
            self.total_accessed += 1

        self.logger.debug(f"🔍 检索工作记忆: {len(result_items)} 项")
        return result_items

    def retrieve_by_type(self,
                        memory_type: MemoryType,
                        limit: int = 20) -> List[WorkingMemoryItem]:
        """
        按类型检索记忆

        Args:
            memory_type: 记忆类型
            limit: 最大返回数量

        Returns:
            记忆项列表
        """
        item_ids = self.type_index.get(memory_type, [])
        items = []

        for item_id in item_ids:
            if item_id in self.items:
                items.append(self.items[item_id])

        # 按访问时间排序（最新的在前）
        items.sort(key=lambda x: x.last_accessed, reverse=True)

        return items[:limit]

    def retrieve_by_tags(self,
                        tags: List[str],
                        match_all: bool = False) -> List[WorkingMemoryItem]:
        """
        按标签检索记忆

        Args:
            tags: 标签列表
            match_all: 是否匹配所有标签

        Returns:
            记忆项列表
        """
        if not tags:
            return []

        # 获取包含这些标签的项目ID
        candidate_sets = []
        for tag in tags:
            if tag in self.tag_index:
                candidate_sets.append(set(self.tag_index[tag]))

        if not candidate_sets:
            return []

        # 交集或并集
        if match_all:
            matching_ids = set.intersection(*candidate_sets)
        else:
            matching_ids = set.union(*candidate_sets)

        # 获取项目
        items = []
        for item_id in matching_ids:
            if item_id in self.items:
                items.append(self.items[item_id])

        # 按相关性排序
        items.sort(key=lambda x: x.relevance_score, reverse=True)

        return items

    def update_attention(self, item_id: str, new_attention: AttentionLevel):
        """
        更新项目的注意力等级

        Args:
            item_id: 项目ID
            new_attention: 新的注意力等级
        """
        if item_id not in self.items:
            self.logger.warning(f"⚠️ 尝试更新不存在的项目: {item_id}")
            return

        item = self.items[item_id]
        old_attention = item.attention

        # 从旧索引中移除
        if item_id in self.attention_index[old_attention]:
            self.attention_index[old_attention].remove(item_id)

        # 更新注意力
        item.attention = new_attention

        # 添加到新索引
        self.attention_index[new_attention].append(item_id)

        self.logger.debug(f"🎯 更新注意力: {item_id[:8]} {old_attention.value} -> {new_attention.value}")

    def remove_item(self, item_id: str) -> bool:
        """
        从工作记忆中移除项目

        Args:
            item_id: 项目ID

        Returns:
            是否成功移除
        """
        if item_id not in self.items:
            return False

        item = self.items[item_id]

        # 从各个索引中移除
        if item_id in self.attention_index.get(item.attention, []):
            self.attention_index[item.attention].remove(item_id)

        if item_id in self.type_index.get(item.memory_type, []):
            self.type_index[item.memory_type].remove(item_id)

        for tag in item.tags:
            if tag in self.tag_index and item_id in self.tag_index[tag]:
                self.tag_index[tag].remove(item_id)

        # 从插入顺序中移除
        if item_id in self.insertion_order:
            self.insertion_order.remove(item_id)

        # 删除项目
        del self.items[item_id]
        self.total_removed += 1

        self.logger.debug(f"🗑️ 从工作记忆移除: {item_id[:8]}")
        return True

    def _evict_least_important(self):
        """淘汰最不重要的项目"""
        if not self.items:
            return

        # 计算每个项目的优先级分数（越低越容易被淘汰）
        candidates = []
        for item_id, item in self.items.items():
            # 锁定的项目不能被淘汰
            if item.is_locked:
                continue

            score = 0.0

            # 注意力等级评分
            attention_scores = {
                AttentionLevel.FOCUSED: 1.0,
                AttentionLevel.ACTIVE: 0.7,
                AttentionLevel.MAINTAINED: 0.4,
                AttentionLevel.PERIPHERAL: 0.1
            }
            score += attention_scores.get(item.attention, 0.0) * 0.4

            # 相关性评分
            score += item.relevance_score * 0.3

            # 访问频率评分
            frequency_score = min(item.access_count * 0.05, 0.3)
            score += frequency_score * 0.2

            # 最近访问加分
            time_since_access = (datetime.now() - item.last_accessed).total_seconds()
            recency_score = max(0, 1 - time_since_access / 3600)  # 1小时内有效
            score += recency_score * 0.1

            candidates.append((score, item_id))

        if not candidates:
            self.logger.warning("⚠️ 所有项目都被锁定，无法淘汰")
            return

        # 选择分数最低的项目
        candidates.sort(key=lambda x: x[0])
        evict_id = candidates[0][1]

        self.remove_item(evict_id)
        self.logger.debug(f"📤 淘汰项目: {evict_id[:8]}")

    def _generate_item_id(self, content: Any) -> str:
        """
        生成项目ID

        Args:
            content: 内容

        Returns:
            唯一ID
        """
        timestamp = datetime.now().isoformat()
        content_str = str(content)[:100] if content else ""
        raw_string = f"{timestamp}_{content_str}_{id(content)}"
        return hashlib.md5(raw_string.encode()).hexdigest()

    def create_thought_chain(self, chain_id: Optional[str] = None) -> ThoughtChain:
        """
        创建思维链

        Args:
            chain_id: 思维链ID（可选，自动生成）

        Returns:
            思维链对象
        """
        if chain_id is None:
            chain_id = f"chain_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"

        chain = ThoughtChain(chain_id=chain_id)
        self.active_chains[chain_id] = chain

        self.logger.debug(f"🧠 创建思维链: {chain_id}")
        return chain

    def get_active_chain(self, chain_id: str) -> Optional[ThoughtChain]:
        """
        获取活跃的思维链

        Args:
            chain_id: 思维链ID

        Returns:
            思维链对象或None
        """
        return self.active_chains.get(chain_id)

    def complete_thought_chain(self, chain_id: str, conclusion: str) -> Optional[ThoughtChain]:
        """
        完成思维链

        Args:
            chain_id: 思维链ID
            conclusion: 结论

        Returns:
            完成的思维链或None
        """
        if chain_id not in self.active_chains:
            self.logger.warning(f"⚠️ 思维链不存在: {chain_id}")
            return None

        chain = self.active_chains[chain_id]
        chain.complete(conclusion)

        # 移动到已完成列表
        del self.active_chains[chain_id]
        self.completed_chains.append(chain)

        self.logger.info(f"✅ 完成思维链: {chain_id}, 步骤数: {len(chain.steps)}")
        return chain

    def update_context(self, **kwargs):
        """
        更新当前上下文

        Args:
            **kwargs: 上下文字段和值
        """
        for key, value in kwargs.items():
            if key in self.current_context:
                self.current_context[key] = value
            else:
                self.logger.warning(f"⚠️ 未知的上下文字段: {key}")

        self.logger.debug(f"📝 更新上下文: {list(kwargs.keys())}")

    def get_context_summary(self) -> Dict[str, Any]:
        """
        获取上下文摘要

        Returns:
            上下文信息字典
        """
        summary = {
            "task_description": self.current_context.get("task_description", ""),
            "user_intent": self.current_context.get("user_intent"),
            "conversation_stage": self.current_context.get("conversation_stage"),
            "recent_topics": list(self.current_context.get("recent_topics", [])),
            "active_items_count": len(self.items),
            "active_chains_count": len(self.active_chains),
            "capacity_usage": len(self.items) / self.capacity if self.capacity > 0 else 0
        }
        return summary

    def add_relation(self, item_id_1: str, item_id_2: str):
        """
        添加两个项目之间的关联

        Args:
            item_id_1: 第一个项目ID
            item_id_2: 第二个项目ID
        """
        if item_id_1 not in self.items or item_id_2 not in self.items:
            self.logger.warning(f"⚠️ 尝试关联不存在的项目")
            return

        item1 = self.items[item_id_1]
        item2 = self.items[item_id_2]

        if item_id_2 not in item1.related_items:
            item1.related_items.append(item_id_2)

        if item_id_1 not in item2.related_items:
            item2.related_items.append(item_id_1)

        self.logger.debug(f"🔗 添加关联: {item_id_1[:8]} <-> {item_id_2[:8]}")

    def get_related_items(self, item_id: str, limit: int = 5) -> List[WorkingMemoryItem]:
        """
        获取相关项目

        Args:
            item_id: 项目ID
            limit: 最大返回数量

        Returns:
            相关项目列表
        """
        if item_id not in self.items:
            return []

        item = self.items[item_id]
        related_items = []

        for related_id in item.related_items:
            if related_id in self.items:
                related_items.append(self.items[related_id])

        # 按访问时间排序
        related_items.sort(key=lambda x: x.last_accessed, reverse=True)

        return related_items[:limit]

    def lock_item(self, item_id: str):
        """
        锁定项目（防止被淘汰）

        Args:
            item_id: 项目ID
        """
        if item_id in self.items:
            self.items[item_id].is_locked = True
            self.logger.debug(f"🔒 锁定项目: {item_id[:8]}")

    def unlock_item(self, item_id: str):
        """
        解锁项目

        Args:
            item_id: 项目ID
        """
        if item_id in self.items:
            self.items[item_id].is_locked = False
            self.logger.debug(f"🔓 解锁项目: {item_id[:8]}")

    def cleanup_expired(self, max_age_hours: float = 2.0):
        """
        清理过期项目

        Args:
            max_age_hours: 最大存活时间（小时）
        """
        now = datetime.now()
        expired_ids = []

        for item_id, item in self.items.items():
            if item.is_locked:
                continue

            age = (now - item.created_at).total_seconds() / 3600

            # 根据注意力等级设置不同的过期时间
            max_age = max_age_hours
            if item.attention == AttentionLevel.FOCUSED:
                max_age *= 2
            elif item.attention == AttentionLevel.ACTIVE:
                max_age *= 1.5
            elif item.attention == AttentionLevel.PERIPHERAL:
                max_age *= 0.5

            if age > max_age:
                expired_ids.append(item_id)

        for item_id in expired_ids:
            self.remove_item(item_id)

        if expired_ids:
            self.logger.info(f"🧹 清理过期项目: {len(expired_ids)} 个")

        self.last_cleanup = now

    def decay_all(self, decay_rate: float = 0.01):
        """
        对所有项目进行衰减处理

        Args:
            decay_rate: 衰减速率
        """
        for item in self.items.values():
            if item.is_locked:
                continue

            # 增加衰减计数
            item.decay_counter += decay_rate

            # 根据衰减程度降低相关性
            if item.decay_counter > 0.5:
                item.relevance_score = max(0.1, item.relevance_score - 0.05)

            # 严重衰减时降低注意力
            if item.decay_counter > 1.0 and item.attention != AttentionLevel.PERIPHERAL:
                item.decrease_attention()
                item.decay_counter = 0.0

    def promote_to_longterm(self, item_id: str) -> Optional[WorkingMemoryItem]:
        """
        提升项目到长期记忆（标记为可转移）

        Args:
            item_id: 项目ID

        Returns:
            被提升的项目或None
        """
        if item_id not in self.items:
            return None

        item = self.items[item_id]

        # 标记元数据
        item.metadata["promote_to_longterm"] = True
        item.metadata["promote_time"] = datetime.now().isoformat()

        self.total_promoted_to_longterm += 1

        self.logger.info(f"⬆️ 提升到长期记忆: {item_id[:8]}, 类型={item.memory_type.value}")
        return item

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取工作记忆统计信息

        Returns:
            统计信息字典
        """
        # 按注意力等级统计
        attention_stats = {}
        for level in AttentionLevel:
            attention_stats[level.value] = len(self.attention_index.get(level, []))

        # 按类型统计
        type_stats = {}
        for mtype in MemoryType:
            type_stats[mtype.value] = len(self.type_index.get(mtype, []))

        stats = {
            "total_items": len(self.items),
            "capacity": self.capacity,
            "usage_percentage": round(len(self.items) / self.capacity * 100, 2) if self.capacity > 0 else 0,
            "attention_distribution": attention_stats,
            "type_distribution": type_stats,
            "active_chains": len(self.active_chains),
            "completed_chains": len(self.completed_chains),
            "total_added": self.total_added,
            "total_removed": self.total_removed,
            "total_accessed": self.total_accessed,
            "total_promoted": self.total_promoted_to_longterm,
            "locked_items": sum(1 for item in self.items.values() if item.is_locked),
            "last_cleanup": self.last_cleanup.isoformat()
        }

        return stats

    def get_focused_items(self) -> List[WorkingMemoryItem]:
        """
        获取所有高度关注的项目

        Returns:
            高度关注的项目列表
        """
        focused_ids = self.attention_index.get(AttentionLevel.FOCUSED, [])
        items = []

        for item_id in focused_ids:
            if item_id in self.items:
                items.append(self.items[item_id])

        return items

    def clear_all(self):
        """清空所有工作记忆"""
        self.items.clear()
        self.insertion_order.clear()

        for level in self.attention_index:
            self.attention_index[level].clear()

        for mtype in self.type_index:
            self.type_index[mtype].clear()

        self.tag_index.clear()
        self.active_chains.clear()

        self.logger.info("🧹 工作记忆已清空")

    def __len__(self) -> int:
        """返回工作记忆中的项目数量"""
        return len(self.items)

    def __contains__(self, item_id: str) -> bool:
        """检查项目是否存在于工作记忆中"""
        return item_id in self.items

    def __repr__(self) -> str:
        """字符串表示"""
        return f"WorkingMemory(items={len(self.items)}, capacity={self.capacity}, chains={len(self.active_chains)})"
