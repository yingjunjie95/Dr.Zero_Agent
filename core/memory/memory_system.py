"""
记忆系统 - Memory System
统一的记忆管理接口，协调三层记忆架构和记忆折叠
- 感官记忆：快速捕获和短暂存储
- 工作记忆：当前任务的活跃信息处理
- 长期记忆：持久化知识和经验存储
- 记忆折叠：智能压缩和知识提炼
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from .sensory_memory import SensoryMemory, SensoryModality, ImportanceLevel
from .working_memory import WorkingMemory, MemoryType, AttentionLevel
from .long_term_memory import LongTermMemory, MemoryCategory
from .memory_folding import MemoryFoldingEngine


class MemorySystem:
    """
    统一记忆系统

    功能：
    - 三层记忆协调：管理感官、工作、长期记忆的交互
    - 记忆流转：自动将重要信息从低层提升到高层
    - 智能检索：跨层级检索相关记忆
    - 记忆维护：定期清理、折叠、巩固
    - 上下文管理：维护当前对话和任务上下文

    设计原则：
    - 透明性：上层应用无需关心记忆存储细节
    - 自动化：记忆流转和维护自动进行
    - 可扩展：支持新的记忆类型和处理策略
    - 高效性：优化的检索和存储性能
    """

    def __init__(self,
                 max_sensory_size: int = 100,
                 working_memory_capacity: int = 50,
                 long_term_capacity: int = 10000,
                 embedding_dim: int = 768,
                 storage_path: str = "data/long_term_memory"):
        """
        初始化记忆系统

        Args:
            max_sensory_size: 感官记忆容量
            working_memory_capacity: 工作记忆容量
            long_term_capacity: 长期记忆容量
            embedding_dim: 嵌入维度
            storage_path: 长期记忆存储路径
        """
        self.logger = logging.getLogger("MemorySystem")

        # 初始化三层记忆
        self.sensory_memory = SensoryMemory(
            max_capacity=max_sensory_size,
            default_ttl_seconds=300
        )

        self.working_memory = WorkingMemory(
            capacity=working_memory_capacity,
            embedding_dim=embedding_dim
        )

        self.long_term_memory = LongTermMemory(
            storage_path=storage_path,
            max_capacity=long_term_capacity
        )

        # 记忆折叠引擎
        self.folding_engine = MemoryFoldingEngine()

        # 当前上下文
        self.current_context = {
            "task_description": "",
            "user_intent": None,
            "conversation_stage": "initial",
            "recent_topics": [],
            "active_goals": []
        }

        # 统计信息
        self.total_interactions = 0
        self.total_promotions = 0  # 记忆提升次数

        self.logger.info(f"✅ 记忆系统初始化完成")
        self.logger.info(f"   - 感官记忆: {max_sensory_size}")
        self.logger.info(f"   - 工作记忆: {working_memory_capacity}")
        self.logger.info(f"   - 长期记忆: {long_term_capacity}")

    def store_sensory(self,
                     content: Any,
                     modality: SensoryModality = SensoryModality.TEXT,
                     source: str = "user",
                     importance: Optional[ImportanceLevel] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        存储感官记忆

        Args:
            content: 内容
            modality: 模态类型
            source: 来源
            importance: 重要性等级
            metadata: 元数据

        Returns:
            项目ID
        """
        item_id = self.sensory_memory.store(
            content=content,
            modality=modality,
            source=source,
            importance=importance,
            metadata=metadata
        )

        if item_id:
            self.logger.debug(f"📥 感官记忆存储: {item_id[:8]}")

        return item_id

    def promote_to_working(self,
                          sensory_item_id: str,
                          attention: AttentionLevel = AttentionLevel.ACTIVE,
                          relevance_score: float = 0.5,
                          tags: Optional[List[str]] = None) -> Optional[str]:
        """
        将感官记忆提升到工作记忆

        Args:
            sensory_item_id: 感官记忆ID
            attention: 注意力等级
            relevance_score: 相关性得分
            tags: 标签

        Returns:
            工作记忆项ID
        """
        # 从感官记忆获取项目
        sensory_item = self.sensory_memory.promote_to_working(sensory_item_id)

        if not sensory_item:
            self.logger.warning(f"⚠️ 无法提升感官记忆: {sensory_item_id}")
            return None

        # 添加到工作记忆
        working_id = self.working_memory.add_item(
            content=sensory_item.content,
            memory_type=MemoryType.WORKING,
            attention=attention,
            relevance_score=relevance_score,
            tags=tags or [],
            source_memory_id=sensory_item_id
        )

        self.total_promotions += 1
        self.logger.info(f"⬆️ 提升到工作记忆: {sensory_item_id[:8]} -> {working_id[:8]}")

        return working_id

    def consolidate_to_longterm(self,
                               working_item_id: str,
                               category: MemoryCategory = MemoryCategory.EPISODIC,
                               importance: float = 0.5) -> Optional[str]:
        """
        将工作记忆巩固到长期记忆

        Args:
            working_item_id: 工作记忆ID
            category: 记忆分类
            importance: 重要性

        Returns:
            长期记忆ID
        """
        if working_item_id not in self.working_memory:
            self.logger.warning(f"⚠️ 工作记忆不存在: {working_item_id}")
            return None

        working_item = self.working_memory.items[working_item_id]

        # 标记为可转移到长期记忆
        promoted_item = self.working_memory.promote_to_longterm(working_item_id)

        if not promoted_item:
            return None

        # 存储到长期记忆
        longterm_id = self.long_term_memory.store(
            content=str(working_item.content),
            category=category,
            importance=importance,
            tags=working_item.tags,
            metadata=working_item.metadata,
            source_context=f"working_memory_{working_item_id}"
        )

        if longterm_id:
            self.logger.info(f"💎 巩固到长期记忆: {working_item_id[:8]} -> {longterm_id[:8]}")

        return longterm_id

    def retrieve_relevant_knowledge(self,
                                   query: str,
                                   context: Optional[Dict[str, Any]] = None,
                                   top_k: int = 10) -> List[Dict[str, Any]]:
        """
        检索相关知识（跨层级检索）

        Args:
            query: 查询文本
            context: 上下文信息
            top_k: 返回数量

        Returns:
            相关记忆列表
        """
        results = []

        # 1. 从工作记忆检索（高优先级）
        working_items = self.working_memory.retrieve_by_relevance(
            query=query,
            top_k=top_k // 2
        )

        for item in working_items:
            results.append({
                "source": "working_memory",
                "item_id": item.item_id,
                "content": item.content,
                "relevance": item.relevance_score,
                "attention": item.attention.value,
                "type": item.memory_type.value
            })

        # 2. 从长期记忆检索
        remaining_k = top_k - len(results)
        if remaining_k > 0:
            longterm_items = self.long_term_memory.retrieve_by_query(
                query=query,
                top_k=remaining_k
            )

            for item in longterm_items:
                results.append({
                    "source": "long_term_memory",
                    "item_id": item.item_id,
                    "content": item.content,
                    "relevance": item.importance,
                    "category": item.category.value,
                    "strength": item.calculate_strength()
                })

        # 按相关性排序
        results.sort(key=lambda x: x.get("relevance", 0), reverse=True)

        self.logger.debug(f"🔍 检索相关知识: {len(results)} 条结果")

        return results

    def store_interaction(self,
                         user_input: str,
                         response: str,
                         context: Optional[Dict[str, Any]] = None,
                         outcome: str = "success"):
        """
        存储完整交互记录

        Args:
            user_input: 用户输入
            response: Agent响应
            context: 上下文
            outcome: 结果（success/failure）
        """
        self.total_interactions += 1

        # 1. 存储到感官记忆
        sensory_id = self.store_sensory(
            content=user_input,
            modality=SensoryModality.TEXT,
            source="user",
            importance=ImportanceLevel.HIGH
        )

        # 2. 提升到工作记忆
        if sensory_id:
            working_id = self.promote_to_working(
                sensory_item_id=sensory_id,
                attention=AttentionLevel.FOCUSED,
                relevance_score=0.8,
                tags=["interaction", "user_input"]
            )

            # 3. 如果交互成功且重要，巩固到长期记忆
            if working_id and outcome == "success":
                # 评估重要性
                importance = self._evaluate_interaction_importance(
                    user_input, response, outcome
                )

                if importance >= 0.6:
                    self.consolidate_to_longterm(
                        working_item_id=working_id,
                        category=MemoryCategory.EPISODIC,
                        importance=importance
                    )

        # 4. 更新上下文
        if context:
            self.update_context(**context)

        self.logger.debug(f"💾 交互记录已存储 (第{self.total_interactions}次)")

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
                self.current_context[key] = value

        # 同步到工作记忆
        self.working_memory.update_context(**kwargs)

        self.logger.debug(f"📝 上下文已更新: {list(kwargs.keys())}")

    def get_current_context(self) -> Dict[str, Any]:
        """获取当前上下文"""
        return self.current_context.copy()

    def needs_folding(self) -> bool:
        """判断是否需要执行记忆折叠"""
        return self.folding_engine.needs_folding()

    def perform_memory_folding(self) -> int:
        """
        执行记忆折叠

        Returns:
            折叠的记忆组数量
        """
        try:
            # 从长期记忆获取候选记忆
            all_memories = list(self.long_term_memory.memories.values())

            if len(all_memories) < self.folding_engine.min_group_size:
                self.logger.debug("ℹ️ 记忆数量不足，跳过折叠")
                return 0

            # 转换为字典格式
            memory_dicts = [
                {
                    "item_id": mem.item_id,
                    "content": mem.content,
                    "tags": mem.tags,
                    "created_at": mem.created_at.isoformat(),
                    "timestamp": mem.created_at.isoformat()
                }
                for mem in all_memories
            ]

            # 检测折叠候选
            candidates = self.folding_engine.detect_folding_candidates(memory_dicts)

            if not candidates:
                self.logger.debug("ℹ️ 未检测到可折叠的记忆组")
                return 0

            # 执行折叠
            folded_count = 0
            memory_contents = {mem["item_id"]: mem["content"] for mem in memory_dicts}

            for candidate in candidates[:5]:  # 限制每次折叠的组数
                folded_memory = self.folding_engine.execute_folding(
                    candidate=candidate,
                    memory_contents=memory_contents
                )

                if folded_memory:
                    folded_count += 1

                    # 将折叠记忆存储到长期记忆
                    self.long_term_memory.store(
                        content=folded_memory.folded_content,
                        category=MemoryCategory.SEMANTIC,
                        importance=0.7,
                        tags=folded_memory.tags + ["folded"],
                        metadata={
                            "folded_id": folded_memory.folded_id,
                            "original_count": len(folded_memory.original_memory_ids),
                            "strategy": folded_memory.folding_strategy.value
                        }
                    )

            if folded_count > 0:
                self.logger.info(f"✨ 记忆折叠完成: {folded_count} 组")

            return folded_count

        except Exception as e:
            self.logger.error(f"❌ 记忆折叠失败: {str(e)}")
            return 0

    def clean_expired_memories(self) -> int:
        """
        清理过期记忆

        Returns:
            清理的记忆数量
        """
        cleaned = 0

        # 清理感官记忆
        sensory_cleaned = self.sensory_memory.cleanup()
        cleaned += sensory_cleaned

        # 清理工作记忆（过期项目）
        self.working_memory.cleanup_expired(max_age_hours=2.0)

        # 清理长期记忆（遗忘机制）
        longterm_forgotten = self.long_term_memory.forget_memories()
        cleaned += longterm_forgotten

        # 清理低质量折叠
        low_quality_cleaned = self.folding_engine.cleanup_low_quality_foldings()

        if cleaned > 0:
            self.logger.info(f"🧹 清理过期记忆: {cleaned} 条")

        return cleaned

    def get_statistics(self) -> Dict[str, Any]:
        """获取完整的记忆系统统计信息"""
        return {
            "total_interactions": self.total_interactions,
            "total_promotions": self.total_promotions,
            "sensory_memory": self.sensory_memory.get_statistics(),
            "working_memory": self.working_memory.get_statistics(),
            "long_term_memory": self.long_term_memory.get_statistics(),
            "memory_folding": self.folding_engine.get_statistics(),
            "current_context": self.current_context
        }

    def get_capacity_description(self) -> str:
        """获取记忆容量描述"""
        sensory_stats = self.sensory_memory.get_statistics()
        working_stats = self.working_memory.get_statistics()
        longterm_stats = self.long_term_memory.get_statistics()

        return (
            f"感官:{sensory_stats['current_items']}/{sensory_stats['total_capacity']} | "
            f"工作:{working_stats['total_items']}/{working_stats['capacity']} | "
            f"长期:{longterm_stats['total_memories']}/{longterm_stats['max_capacity']}"
        )

    def save_to_persistence(self):
        """保存所有记忆到持久化存储"""
        try:
            # 保存长期记忆
            self.long_term_memory.save_to_persistence()

            # 保存折叠引擎状态（如果需要）
            # self.folding_engine.save_state()

            self.logger.info("💾 记忆系统已保存到持久化存储")

        except Exception as e:
            self.logger.error(f"❌ 保存记忆系统失败: {str(e)}")

    def clear_caches(self):
        """清理缓存"""
        # 清理感官记忆
        self.sensory_memory.clear()

        # 清理工作记忆中的边缘项目
        peripheral_items = self.working_memory.attention_index.get(
            AttentionLevel.PERIPHERAL, []
        )
        for item_id in peripheral_items[:10]:
            self.working_memory.remove_item(item_id)

        self.logger.info("🧹 记忆缓存已清理")

    def reduce_working_memory_capacity(self, reduction_ratio: float = 0.3):
        """
        减少工作记忆容量

        Args:
            reduction_ratio: 减少比例
        """
        current_capacity = self.working_memory.capacity
        new_capacity = max(10, int(current_capacity * (1 - reduction_ratio)))

        # 淘汰最不重要的项目
        items_to_remove = len(self.working_memory) - new_capacity

        if items_to_remove > 0:
            peripheral_items = self.working_memory.attention_index.get(
                AttentionLevel.PERIPHERAL, []
            )
            removed = 0

            for item_id in peripheral_items:
                if removed >= items_to_remove:
                    break
                if self.working_memory.remove_item(item_id):
                    removed += 1

            self.logger.info(
                f"📉 工作记忆容量调整: {current_capacity} -> {new_capacity}, "
                f"移除 {removed} 项"
            )

    def release_low_priority_memories(self):
        """释放低优先级记忆"""
        # 从工作记忆释放边缘记忆
        peripheral_items = self.working_memory.attention_index.get(
            AttentionLevel.PERIPHERAL, []
        ).copy()

        released = 0
        for item_id in peripheral_items:
            if self.working_memory.remove_item(item_id):
                released += 1

        if released > 0:
            self.logger.info(f"🗑️ 释放低优先级记忆: {released} 项")

    def _evaluate_interaction_importance(self,
                                        user_input: str,
                                        response: str,
                                        outcome: str) -> float:
        """
        评估交互的重要性

        Args:
            user_input: 用户输入
            response: Agent响应
            outcome: 结果

        Returns:
            重要性评分 (0.0-1.0)
        """
        importance = 0.5  # 基础重要性

        # 基于结果调整
        if outcome == "success":
            importance += 0.1
        else:
            importance -= 0.2

        # 基于输入长度（较长的输入通常更重要）
        if len(user_input) > 100:
            importance += 0.1

        # 基于关键词
        important_keywords = ["重要", "关键", "记住", "important", "key", "remember"]
        if any(kw in user_input.lower() for kw in important_keywords):
            importance += 0.2

        return max(0.0, min(1.0, importance))
