"""
感官记忆 - Sensory Memory
记忆系统的第一层，负责快速捕获和短暂存储原始感知信息
特性：
- 超短期存储（几秒到几分钟）
- 高容量但快速衰减
- 自动过滤和重要性评估
- 为工作记忆提供筛选后的信息
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import hashlib


class SensoryModality(Enum):
    """感官模态类型"""
    TEXT = "text"
    VISUAL = "visual"
    AUDIO = "audio"
    CONTEXTUAL = "contextual"
    EMOTIONAL = "emotional"
    METADATA = "metadata"


class ImportanceLevel(Enum):
    """重要性等级"""
    CRITICAL = "critical"  # 关键信息，必须保留
    HIGH = "high"  # 高重要性
    MEDIUM = "medium"  # 中等重要性
    LOW = "low"  # 低重要性，可快速丢弃
    TRIVIAL = "trivial"  # 琐碎信息，立即丢弃


@dataclass
class SensoryItem:
    """感官记忆项"""
    item_id: str
    content: Any  # 原始内容
    modality: SensoryModality
    importance: ImportanceLevel
    timestamp: datetime
    source: str  # 信息来源（user_input, system, environment等）

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 生命周期管理
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    # 衰减参数
    decay_rate: float = 0.1  # 衰减速率 (0.0-1.0)
    current_strength: float = 1.0  # 当前强度 (0.0-1.0)

    # 标记
    is_processed: bool = False  # 是否已被处理
    is_promoted: bool = False  # 是否已提升到工作记忆

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "item_id": self.item_id,
            "content": str(self.content)[:200] if self.content else "",  # 限制长度
            "modality": self.modality.value,
            "importance": self.importance.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "current_strength": round(self.current_strength, 3),
            "is_processed": self.is_processed,
            "is_promoted": self.is_promoted
        }

    def access(self):
        """访问该项目，更新访问记录"""
        self.last_accessed = datetime.now()
        self.access_count += 1
        # 访问可以稍微增强强度
        self.current_strength = min(1.0, self.current_strength + 0.05)

    def decay(self, time_delta_seconds: float):
        """
        根据时间衰减强度

        Args:
            time_delta_seconds: 经过的秒数
        """
        # 基于重要性的衰减速率调整
        importance_decay_modifier = {
            ImportanceLevel.CRITICAL: 0.1,
            ImportanceLevel.HIGH: 0.3,
            ImportanceLevel.MEDIUM: 0.6,
            ImportanceLevel.LOW: 1.0,
            ImportanceLevel.TRIVIAL: 2.0
        }

        modifier = importance_decay_modifier.get(self.importance, 1.0)
        decay_amount = self.decay_rate * modifier * (time_delta_seconds / 60.0)

        self.current_strength = max(0.0, self.current_strength - decay_amount)

    def is_expired(self, min_strength: float = 0.1) -> bool:
        """检查是否已过期"""
        return self.current_strength < min_strength


@dataclass
class SensorySnapshot:
    """感官快照 - 某一时刻的完整感官状态"""
    snapshot_id: str
    timestamp: datetime
    items: List[SensoryItem]
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp.isoformat(),
            "items_count": len(self.items),
            "items": [item.to_dict() for item in self.items],
            "context": self.context
        }


class SensoryMemory:
    """
    感官记忆系统

    功能：
    - 快速捕获：接收来自环境的原始感知信息
    - 自动过滤：基于重要性评估过滤信息
    - 短暂存储：维持几秒到几分钟的短期存储
    - 智能衰减：根据重要性和使用情况动态衰减
    - 选择性提升：将重要信息传递到工作记忆

    设计原则：
    - 高性能：支持快速读写操作
    - 自动管理：无需手动清理，自动衰减和移除
    - 容量控制：通过强度和数量双重限制
    """

    def __init__(self, max_capacity: int = 100, default_ttl_seconds: int = 300):
        """
        初始化感官记忆

        Args:
            max_capacity: 最大容量（项目数量）
            default_ttl_seconds: 默认生存时间（秒）
        """
        self.max_capacity = max_capacity
        self.default_ttl_seconds = default_ttl_seconds
        self.logger = logging.getLogger("SensoryMemory")

        # 记忆存储 - 使用双端队列便于快速移除过期项目
        self.items: deque = deque(maxlen=max_capacity)

        # 索引 - 加速查找
        self.id_index: Dict[str, SensoryItem] = {}
        self.source_index: Dict[str, List[str]] = {}  # source -> [item_ids]
        self.modality_index: Dict[SensoryModality, List[str]] = {
            mod: [] for mod in SensoryModality
        }

        # 快照历史
        self.snapshots: deque = deque(maxlen=50)

        # 统计信息
        self.total_received = 0
        self.total_filtered = 0
        self.total_promoted = 0
        self.total_decayed = 0

        # 最后清理时间
        self.last_cleanup = datetime.now()

        self.logger.info(f"✅ 感官记忆初始化完成 - 容量: {max_capacity}, TTL: {default_ttl_seconds}秒")

    def store(self,
              content: Any,
              modality: SensoryModality = SensoryModality.TEXT,
              source: str = "user",
              importance: Optional[ImportanceLevel] = None,
              metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        存储感官信息

        Args:
            content: 内容
            modality: 模态类型
            source: 来源
            importance: 重要性等级（None则自动评估）
            metadata: 元数据

        Returns:
            项目ID，如果被过滤则返回None
        """
        try:
            # 自动评估重要性（如果未指定）
            if importance is None:
                importance = self._assess_importance(content, modality, source)

            # 过滤琐碎信息
            if importance == ImportanceLevel.TRIVIAL:
                self.total_filtered += 1
                self.logger.debug(f"🚫 过滤琐碎信息: {str(content)[:50]}")
                return None

            # 创建项目ID
            item_id = self._generate_item_id(content, source)

            # 创建感官项目
            item = SensoryItem(
                item_id=item_id,
                content=content,
                modality=modality,
                importance=importance,
                timestamp=datetime.now(),
                source=source,
                metadata=metadata or {},
                decay_rate=self._calculate_decay_rate(importance)
            )

            # 添加到存储
            self.items.append(item)

            # 更新索引
            self.id_index[item_id] = item
            self._update_source_index(source, item_id)
            self._update_modality_index(modality, item_id)

            # 更新统计
            self.total_received += 1

            self.logger.debug(f"💾 存储感官记忆: ID={item_id[:8]}, 重要性={importance.value}")

            # 定期清理
            if self._should_cleanup():
                self.cleanup()

            return item_id

        except Exception as e:
            self.logger.error(f"❌ 存储感官记忆失败: {str(e)}")
            return None

    def retrieve_recent(self,
                       count: int = 10,
                       modality: Optional[SensoryModality] = None,
                       min_importance: Optional[ImportanceLevel] = None,
                       source: Optional[str] = None) -> List[SensoryItem]:
        """
        检索最近的感官记忆

        Args:
            count: 返回数量
            modality: 模态过滤
            min_importance: 最低重要性
            source: 来源过滤

        Returns:
            感官项目列表
        """
        # 先执行衰减更新
        self._update_all_decay()

        # 过滤项目
        filtered_items = []
        for item in reversed(self.items):  # 从最新开始
            # 应用过滤条件
            if modality and item.modality != modality:
                continue

            if min_importance and not self._meets_importance_threshold(item.importance, min_importance):
                continue

            if source and item.source != source:
                continue

            # 跳过已过期的
            if item.is_expired():
                continue

            filtered_items.append(item)

            if len(filtered_items) >= count:
                break

        # 标记为已访问
        for item in filtered_items:
            item.access()

        self.logger.debug(f"🔍 检索感官记忆: {len(filtered_items)} 项")
        return filtered_items

    def retrieve_by_timeframe(self,
                             seconds_ago: float,
                             modality: Optional[SensoryModality] = None) -> List[SensoryItem]:
        """
        检索指定时间范围内的记忆

        Args:
            seconds_ago: 多少秒前到现在
            modality: 模态过滤

        Returns:
            感官项目列表
        """
        cutoff_time = datetime.now() - timedelta(seconds=seconds_ago)

        items = []
        for item in self.items:
            if item.timestamp >= cutoff_time and not item.is_expired():
                if modality is None or item.modality == modality:
                    items.append(item)
                    item.access()

        # 按时间排序（最新的在前）
        items.sort(key=lambda x: x.timestamp, reverse=True)

        return items

    def promote_to_working(self, item_id: str) -> Optional[SensoryItem]:
        """
        将感官记忆提升到工作记忆

        Args:
            item_id: 项目ID

        Returns:
            被提升的项目，如果不存在则返回None
        """
        if item_id not in self.id_index:
            self.logger.warning(f"⚠️ 尝试提升不存在的项目: {item_id}")
            return None

        item = self.id_index[item_id]

        # 标记为已提升
        item.is_promoted = True
        item.access()

        # 提升时增强强度
        item.current_strength = min(1.0, item.current_strength + 0.2)

        self.total_promoted += 1

        self.logger.info(f"⬆️ 提升到工作记忆: {item_id[:8]}")

        return item

    def get_snapshot(self) -> SensorySnapshot:
        """获取当前感官状态的快照"""
        # 先更新衰减
        self._update_all_decay()

        # 获取所有未过期的项目
        active_items = [item for item in self.items if not item.is_expired()]

        snapshot = SensorySnapshot(
            snapshot_id=self._generate_snapshot_id(),
            timestamp=datetime.now(),
            items=active_items.copy(),
            context={
                "total_items": len(active_items),
                "capacity_usage": len(self.items) / self.max_capacity,
                "modalities": self._count_by_modality(active_items)
            }
        )

        # 保存快照
        self.snapshots.append(snapshot)

        return snapshot

    def cleanup(self) -> int:
        """
        清理过期和低强度的项目

        Returns:
            清理的项目数量
        """
        removed_count = 0
        current_time = datetime.now()

        # 收集需要移除的项目ID
        items_to_remove = []

        for item in self.items:
            # 计算时间差
            time_delta = (current_time - item.last_accessed).total_seconds()

            # 更新衰减
            item.decay(time_delta)

            # 检查是否过期
            if item.is_expired():
                items_to_remove.append(item.item_id)

        # 移除过期项目
        for item_id in items_to_remove:
            if self._remove_item(item_id):
                removed_count += 1
                self.total_decayed += 1

        # 如果仍然超出容量，移除最低强度的项目
        if len(self.items) > self.max_capacity * 0.9:  # 达到90%容量时触发
            excess_count = len(self.items) - int(self.max_capacity * 0.7)  # 清理到70%
            if excess_count > 0:
                removed_count += self._remove_lowest_strength_items(excess_count)

        self.last_cleanup = current_time

        if removed_count > 0:
            self.logger.info(f"🧹 清理感官记忆: {removed_count} 项")

        return removed_count

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 更新衰减
        self._update_all_decay()

        active_items = [item for item in self.items if not item.is_expired()]

        # 按重要性统计
        importance_dist = {}
        for item in active_items:
            imp = item.importance.value
            importance_dist[imp] = importance_dist.get(imp, 0) + 1

        # 按模态统计
        modality_dist = self._count_by_modality(active_items)

        # 平均强度
        avg_strength = sum(item.current_strength for item in active_items) / len(active_items) if active_items else 0.0

        return {
            "total_capacity": self.max_capacity,
            "current_items": len(active_items),
            "capacity_usage": round(len(active_items) / self.max_capacity, 3),
            "total_received": self.total_received,
            "total_filtered": self.total_filtered,
            "total_promoted": self.total_promoted,
            "total_decayed": self.total_decayed,
            "average_strength": round(avg_strength, 3),
            "importance_distribution": importance_dist,
            "modality_distribution": modality_dist,
            "snapshots_count": len(self.snapshots),
            "last_cleanup": self.last_cleanup.isoformat()
        }

    def clear(self):
        """清空所有感官记忆"""
        self.items.clear()
        self.id_index.clear()
        self.source_index.clear()
        for mod in self.modality_index:
            self.modality_index[mod].clear()

        self.logger.info("🔄 感官记忆已清空")

    def _assess_importance(self,
                          content: Any,
                          modality: SensoryModality,
                          source: str) -> ImportanceLevel:
        """
        自动评估信息重要性

        Args:
            content: 内容
            modality: 模态
            source: 来源

        Returns:
            重要性等级
        """
        # 基于来源的初始重要性
        source_importance = {
            "user": ImportanceLevel.HIGH,
            "system": ImportanceLevel.MEDIUM,
            "environment": ImportanceLevel.LOW,
            "internal": ImportanceLevel.LOW
        }

        importance = source_importance.get(source, ImportanceLevel.LOW)

        # 基于内容的调整
        if isinstance(content, str):
            content_lower = content.lower()

            # 关键词检测
            critical_keywords = ["错误", "异常", "失败", "紧急", "critical", "error"]
            high_keywords = ["重要", "注意", "关键", "important", "key"]

            if any(kw in content_lower for kw in critical_keywords):
                importance = ImportanceLevel.CRITICAL
            elif any(kw in content_lower for kw in high_keywords):
                importance = max(importance, ImportanceLevel.HIGH, key=lambda x: list(ImportanceLevel).index(x))

            # 长度因素
            if len(content) > 500:
                importance = max(importance, ImportanceLevel.MEDIUM, key=lambda x: list(ImportanceLevel).index(x))

        # 基于模态的调整
        if modality in [SensoryModality.EMOTIONAL, SensoryModality.CONTEXTUAL]:
            importance = max(importance, ImportanceLevel.MEDIUM, key=lambda x: list(ImportanceLevel).index(x))

        return importance

    def _calculate_decay_rate(self, importance: ImportanceLevel) -> float:
        """根据重要性计算衰减速率"""
        decay_rates = {
            ImportanceLevel.CRITICAL: 0.02,
            ImportanceLevel.HIGH: 0.05,
            ImportanceLevel.MEDIUM: 0.1,
            ImportanceLevel.LOW: 0.2,
            ImportanceLevel.TRIVIAL: 0.5
        }
        return decay_rates.get(importance, 0.1)

    def _generate_item_id(self, content: Any, source: str) -> str:
        """生成唯一的项目ID"""
        timestamp = datetime.now().timestamp()
        content_hash = hashlib.md5(str(content).encode()).hexdigest()[:8]
        return f"sensory_{source}_{content_hash}_{int(timestamp)}"

    def _generate_snapshot_id(self) -> str:
        """生成快照ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"snapshot_{timestamp}"

    def _update_source_index(self, source: str, item_id: str):
        """更新来源索引"""
        if source not in self.source_index:
            self.source_index[source] = []
        self.source_index[source].append(item_id)

    def _update_modality_index(self, modality: SensoryModality, item_id: str):
        """更新模态索引"""
        if item_id not in self.modality_index[modality]:
            self.modality_index[modality].append(item_id)

    def _remove_item(self, item_id: str) -> bool:
        """移除单个项目"""
        if item_id not in self.id_index:
            return False

        item = self.id_index.pop(item_id)

        # 从deque中移除（效率较低，但必要时执行）
        try:
            self.items.remove(item)
        except ValueError:
            pass

        # 从索引中移除
        if item.source in self.source_index:
            if item_id in self.source_index[item.source]:
                self.source_index[item.source].remove(item_id)

        for mod in self.modality_index:
            if item_id in self.modality_index[mod]:
                self.modality_index[mod].remove(item_id)

        return True

    def _remove_lowest_strength_items(self, count: int) -> int:
        """移除强度最低的N个项目"""
        if not self.items:
            return 0

        # 获取所有项目并按强度排序
        sorted_items = sorted(self.items, key=lambda x: x.current_strength)

        removed = 0
        for item in sorted_items[:count]:
            if self._remove_item(item.item_id):
                removed += 1

        return removed

    def _count_by_modality(self, items: List[SensoryItem]) -> Dict[str, int]:
        """按模态统计项目数量"""
        counts = {}
        for item in items:
            mod = item.modality.value
            counts[mod] = counts.get(mod, 0) + 1
        return counts

    def _meets_importance_threshold(self,
                                   item_importance: ImportanceLevel,
                                   min_importance: ImportanceLevel) -> bool:
        """检查是否满足重要性阈值"""
        importance_order = list(ImportanceLevel)
        item_level = importance_order.index(item_importance)
        min_level = importance_order.index(min_importance)
        return item_level >= min_level

    def _update_all_decay(self):
        """更新所有项目的衰减"""
        current_time = datetime.now()
        for item in self.items:
            time_delta = (current_time - item.last_accessed).total_seconds()
            if time_delta > 0:
                item.decay(time_delta)

    def _should_cleanup(self) -> bool:
        """判断是否应该执行清理"""
        # 基于时间（超过60秒）
        time_since_cleanup = (datetime.now() - self.last_cleanup).total_seconds()

        # 基于容量（超过85%）
        capacity_ratio = len(self.items) / self.max_capacity

        return time_since_cleanup > 60 or capacity_ratio > 0.85

    def export_data(self) -> Dict[str, Any]:
        """导出数据用于持久化"""
        return {
            "items": [item.to_dict() for item in self.items if not item.is_expired()],
            "statistics": self.get_statistics(),
            "export_timestamp": datetime.now().isoformat()
        }

    def import_data(self, data: Dict[str, Any]):
        """导入数据"""
        try:
            # 这里可以实现从持久化存储恢复的逻辑
            # 由于感官记忆是短期的，通常不需要持久化
            self.logger.info("ℹ️ 感官记忆通常不持久化，跳过导入")
        except Exception as e:
            self.logger.error(f"❌ 导入感官记忆数据失败: {str(e)}")
