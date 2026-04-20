"""
经验回放 - Experience Replay
存储和重用历史交互经验，支持批量学习和策略优化
实现高效的知识积累和决策改进
专为CPU环境优化，平衡内存使用和 learning 效率
"""
import logging
import random
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
import os


class ExperienceType(Enum):
    """经验类型"""
    SUCCESS = "success"  # 成功经验
    FAILURE = "failure"  # 失败经验
    NEUTRAL = "neutral"  # 中性经验
    INSIGHTFUL = "insightful"  # 有洞察的经验
    CORRECTED = "corrected"  # 被纠正的经验


class PriorityLevel(Enum):
    """优先级等级"""
    LOW = 1  # 低优先级
    MEDIUM = 2  # 中等优先级
    HIGH = 3  # 高优先级
    CRITICAL = 4  # 关键优先级


@dataclass
class Experience:
    """经验记录"""
    experience_id: str
    state: Dict[str, Any]  # 状态（感知、认知等）
    action: Dict[str, Any]  # 行动（决策、工具调用等）
    reward: float  # 奖励信号 (-1.0 to 1.0)
    next_state: Dict[str, Any]  # 下一状态
    done: bool  # 是否结束

    # 元数据
    experience_type: ExperienceType = ExperienceType.NEUTRAL
    priority: PriorityLevel = PriorityLevel.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str = ""
    tags: List[str] = field(default_factory=list)

    # 学习相关
    td_error: float = 0.0  # 时序差分误差（用于优先经验回放）
    sampling_count: int = 0  # 被采样次数
    last_sampled: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "experience_id": self.experience_id,
            "state": self._simplify_state(self.state),
            "action": self._simplify_state(self.action),
            "reward": round(self.reward, 3),
            "next_state": self._simplify_state(self.next_state),
            "done": self.done,
            "experience_type": self.experience_type.value,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "tags": self.tags,
            "td_error": round(self.td_error, 3),
            "sampling_count": self.sampling_count
        }

    @staticmethod
    def _simplify_state(state: Dict[str, Any], max_depth: int = 2) -> Dict[str, Any]:
        """简化状态字典，避免过大"""
        if max_depth <= 0:
            return {"_simplified": True}

        simplified = {}
        for key, value in state.items():
            if isinstance(value, dict):
                simplified[key] = Experience._simplify_state(value, max_depth - 1)
            elif isinstance(value, (str, int, float, bool)):
                # 限制字符串长度
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
class ReplayConfig:
    """回放配置"""
    buffer_size: int = 500  # 经验缓冲区大小
    batch_size: int = 16  # 批量学习大小
    min_experiences: int = 32  # 开始学习的最小经验数
    prioritized_replay: bool = True  # 启用优先经验回放
    priority_alpha: float = 0.6  # 优先级指数（0=均匀采样，1=完全优先）
    priority_beta: float = 0.4  # 重要性采样 beta（用于修正偏差）

    # 衰减和遗忘
    decay_enabled: bool = True  # 启用经验衰减
    decay_rate: float = 0.99  # 衰减速率
    max_age_days: int = 7  # 最大保留天数

    # 持久化
    persistence_enabled: bool = True  # 启用持久化
    save_interval: int = 50  # 保存间隔（每N条经验）
    storage_path: str = "data/experience_replay"

    # 过滤
    min_reward_threshold: float = -0.5  # 最小奖励阈值（过滤低质量经验）
    duplicate_detection: bool = True  # 启用重复检测


class ExperienceReplay:
    """
    经验回放系统

    功能：
    - 经验存储：高效存储和管理历史交互经验
    - 优先采样：基于TD误差的优先经验回放
    - 批量学习：从经验批次中提取模式和改进策略
    - 经验衰减：自动降低旧经验的权重
    - 持久化：保存和加载经验缓冲区
    - 统计分析：提供经验分布和质量指标
    """

    def __init__(self, config: Optional[ReplayConfig] = None):
        """
        初始化经验回放系统

        Args:
            config: 回放配置
        """
        self.config = config or ReplayConfig()
        self.logger = logging.getLogger("ExperienceReplay")

        # 经验缓冲区（使用deque自动管理大小）
        self.buffer: deque = deque(maxlen=self.config.buffer_size)

        # 优先级 SumTree（用于优先经验回放）
        self.priority_tree: List[float] = []
        self.max_priority: float = 1.0

        # 经验索引（快速查找）
        self.experience_index: Dict[str, int] = {}  # experience_id -> buffer index

        # 统计信息
        self.stats = {
            "total_experiences_added": 0,
            "total_batches_sampled": 0,
            "total_learning_updates": 0,
            "experiences_by_type": defaultdict(int),
            "average_reward": 0.0,
            "buffer_utilization": 0.0
        }

        # 创建存储目录
        if self.config.persistence_enabled:
            os.makedirs(self.config.storage_path, exist_ok=True)

        # 加载已有经验
        if self.config.persistence_enabled:
            self._load_from_disk()

        self.logger.info(
            f"✅ 经验回放系统初始化完成 - "
            f"缓冲区大小: {self.config.buffer_size}, "
            f"优先回放: {self.config.prioritized_replay}"
        )

    def add_experience(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
        experience_type: Optional[ExperienceType] = None,
        session_id: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """
        添加经验到缓冲区

        Args:
            state: 当前状态
            action: 采取的行动
            reward: 奖励信号
            next_state: 下一状态
            done: 是否结束
            experience_type: 经验类型
            session_id: 会话ID
            tags: 标签列表

        Returns:
            经验ID
        """
        import uuid

        # 生成唯一ID
        experience_id = f"exp_{uuid.uuid4().hex[:8]}"

        # 确定经验类型
        if experience_type is None:
            experience_type = self._infer_experience_type(reward, done)

        # 确定优先级
        priority = self._calculate_initial_priority(reward, experience_type)

        # 创建经验对象
        experience = Experience(
            experience_id=experience_id,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            experience_type=experience_type,
            priority=priority,
            session_id=session_id,
            tags=tags or []
        )

        # 检查重复
        if self.config.duplicate_detection and self._is_duplicate(experience):
            self.logger.debug(f"⚠️ 检测到重复经验，跳过: {experience_id}")
            return experience_id

        # 添加到缓冲区
        buffer_index = len(self.buffer)
        self.buffer.append(experience)
        self.experience_index[experience_id] = buffer_index

        # 更新优先级树
        if self.config.prioritized_replay:
            self._update_priority_tree(buffer_index, priority ** self.config.priority_alpha)

        # 更新统计
        self.stats["total_experiences_added"] += 1
        self.stats["experiences_by_type"][experience_type.value] += 1
        self._update_average_reward()

        # 定期保存
        if (self.config.persistence_enabled and
            self.stats["total_experiences_added"] % self.config.save_interval == 0):
            self._save_to_disk()

        self.logger.debug(
            f"💾 经验已添加: {experience_id} - "
            f"类型: {experience_type.value}, "
            f"奖励: {reward:.2f}, "
            f"缓冲区大小: {len(self.buffer)}"
        )

        return experience_id

    def sample_batch(self, batch_size: Optional[int] = None) -> Tuple[List[Experience], List[float]]:
        """
        采样一批经验

        Args:
            batch_size: 批次大小，使用配置值如果未指定

        Returns:
            (经验列表, 重要性采样权重列表)
        """
        if len(self.buffer) < self.config.min_experiences:
            self.logger.warning(
                f"⚠️ 经验数量不足: {len(self.buffer)} < {self.config.min_experiences}"
            )
            return [], []

        actual_batch_size = batch_size or self.config.batch_size
        actual_batch_size = min(actual_batch_size, len(self.buffer))

        if self.config.prioritized_replay:
            experiences, weights = self._prioritized_sampling(actual_batch_size)
        else:
            experiences, weights = self._uniform_sampling(actual_batch_size)

        # 更新采样计数
        for exp in experiences:
            exp.sampling_count += 1
            exp.last_sampled = datetime.now()

        self.stats["total_batches_sampled"] += 1

        self.logger.debug(
            f"📊 采样批次: {len(experiences)} 条经验, "
            f"平均权重: {sum(weights)/len(weights):.3f}"
        )

        return experiences, weights

    def update_td_errors(self, td_errors: Dict[str, float]):
        """
        更新TD误差（用于优先经验回放）

        Args:
            td_errors: 经验ID到TD误差的映射
        """
        if not self.config.prioritized_replay:
            return

        for experience_id, td_error in td_errors.items():
            if experience_id in self.experience_index:
                buffer_index = self.experience_index[experience_id]

                # 计算新优先级
                priority = (abs(td_error) + 1e-6) ** self.config.priority_alpha

                # 更新优先级树
                self._update_priority_tree(buffer_index, priority)

                # 更新经验对象
                if buffer_index < len(self.buffer):
                    self.buffer[buffer_index].td_error = td_error

        self.logger.debug(f"🔄 更新了 {len(td_errors)} 条经验的TD误差")

    def get_learning_insights(self) -> Dict[str, Any]:
        """
        获取学习洞察

        Returns:
            学习洞察字典
        """
        if len(self.buffer) == 0:
            return {"message": "经验缓冲区为空"}

        insights = {
            "buffer_size": len(self.buffer),
            "buffer_utilization": len(self.buffer) / self.config.buffer_size,
            "experience_distribution": dict(self.stats["experiences_by_type"]),
            "average_reward": self.stats["average_reward"],
            "reward_distribution": self._calculate_reward_distribution(),
            "temporal_patterns": self._analyze_temporal_patterns(),
            "high_value_experiences": self._find_high_value_experiences(),
            "learning_recommendations": self._generate_learning_recommendations()
        }

        return insights

    def needs_batch_learning(self) -> bool:
        """判断是否需要进行批量学习"""
        return len(self.buffer) >= self.config.min_experiences

    def perform_batch_analysis(self) -> Dict[str, Any]:
        """
        执行批量分析

        Returns:
            分析结果
        """
        if len(self.buffer) < self.config.min_experiences:
            return {"status": "insufficient_experiences"}

        # 采样批次
        experiences, weights = self.sample_batch()

        if not experiences:
            return {"status": "sampling_failed"}

        # 分析批次
        analysis = {
            "batch_size": len(experiences),
            "success_rate": sum(1 for e in experiences if e.reward > 0) / len(experiences),
            "average_reward": sum(e.reward for e in experiences) / len(experiences),
            "common_patterns": self._extract_common_patterns(experiences),
            "improvement_areas": self._identify_improvement_areas(experiences),
            "timestamp": datetime.now().isoformat()
        }

        self.stats["total_learning_updates"] += 1

        self.logger.info(f"📚 批量分析完成 - 成功率: {analysis['success_rate']:.2%}")
        return analysis

    def optimize_strategies(self) -> Dict[str, Any]:
        """
        基于经验优化策略

        Returns:
            优化建议
        """
        insights = self.get_learning_insights()

        optimizations = {
            "strategy_adjustments": [],
            "tool_usage_optimization": [],
            "decision_threshold_tuning": []
        }

        # 基于奖励分布调整策略
        if insights.get("average_reward", 0) < 0:
            optimizations["strategy_adjustments"].append({
                "recommendation": "采用更保守的策略",
                "reason": "平均奖励为负，当前策略效果不佳",
                "priority": "high"
            })

        # 分析成功经验
        high_value = insights.get("high_value_experiences", [])
        if high_value:
            optimizations["strategy_adjustments"].append({
                "recommendation": "强化成功模式",
                "successful_patterns": [exp.get("pattern", "") for exp in high_value[:3]],
                "priority": "medium"
            })

        self.logger.info(f"🎯 策略优化完成 - {len(optimizations['strategy_adjustments'])} 条建议")
        return optimizations

    def clear_old_experiences(self, max_age_days: Optional[int] = None) -> int:
        """
        清理过期经验

        Args:
            max_age_days: 最大保留天数

        Returns:
            清理的经验数量
        """
        if not self.config.decay_enabled:
            return 0

        age_threshold = max_age_days or self.config.max_age_days
        cutoff_time = datetime.now() - timedelta(days=age_threshold)

        old_indices = []
        for i, exp in enumerate(self.buffer):
            if exp.timestamp < cutoff_time:
                old_indices.append(i)

        # 从后往前删除，避免索引变化
        removed_count = 0
        for idx in reversed(old_indices):
            if idx < len(self.buffer):
                exp = self.buffer[idx]
                self.buffer.remove(exp)

                # 更新索引
                if exp.experience_id in self.experience_index:
                    del self.experience_index[exp.experience_id]

                removed_count += 1

        if removed_count > 0:
            self.logger.info(f"🧹 清理了 {removed_count} 条过期经验")
            self._rebuild_index()

        return removed_count

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.config.buffer_size,
            "buffer_utilization": len(self.buffer) / self.config.buffer_size,
            "unique_sessions": len(set(exp.session_id for exp in self.buffer if exp.session_id)),
            "config": {
                "batch_size": self.config.batch_size,
                "prioritized_replay": self.config.prioritized_replay,
                "decay_enabled": self.config.decay_enabled,
                "max_age_days": self.config.max_age_days
            }
        }

    def _infer_experience_type(self, reward: float, done: bool) -> ExperienceType:
        """推断经验类型"""
        if reward > 0.5:
            return ExperienceType.SUCCESS
        elif reward < -0.5:
            return ExperienceType.FAILURE
        elif abs(reward) < 0.1:
            return ExperienceType.NEUTRAL
        elif done and reward > 0:
            return ExperienceType.INSIGHTFUL
        else:
            return ExperienceType.NEUTRAL

    def _calculate_initial_priority(self, reward: float, experience_type: ExperienceType) -> PriorityLevel:
        """计算初始优先级"""
        if experience_type == ExperienceType.SUCCESS or experience_type == ExperienceType.INSIGHTFUL:
            return PriorityLevel.HIGH
        elif experience_type == ExperienceType.FAILURE:
            return PriorityLevel.CRITICAL  # 失败经验更重要，需要学习
        elif abs(reward) > 0.3:
            return PriorityLevel.MEDIUM
        else:
            return PriorityLevel.LOW

    def _is_duplicate(self, new_experience: Experience) -> bool:
        """检查是否是重复经验"""
        # 简化版：检查最近的经验是否有相似的状态和行动
        recent_experiences = list(self.buffer)[-50:]

        for exp in recent_experiences:
            # 比较状态的关键部分
            if (self._states_similar(exp.state, new_experience.state) and
                self._actions_similar(exp.action, new_experience.action)):
                return True

        return False

    @staticmethod
    def _states_similar(state1: Dict, state2: Dict, threshold: float = 0.8) -> bool:
        """判断两个状态是否相似"""
        # 简化实现：比较用户输入
        input1 = state1.get('user_input', '')
        input2 = state2.get('user_input', '')

        if not input1 or not input2:
            return False

        # 简单的相似度检查
        if input1 == input2:
            return True

        # 检查是否包含相同关键词
        words1 = set(input1.lower().split())
        words2 = set(input2.lower().split())

        if not words1 or not words2:
            return False

        intersection = words1 & words2
        union = words1 | words2

        similarity = len(intersection) / len(union) if union else 0
        return similarity > threshold

    @staticmethod
    def _actions_similar(action1: Dict, action2: Dict) -> bool:
        """判断两个行动是否相似"""
        return action1.get('primary_action') == action2.get('primary_action')

    def _update_average_reward(self):
        """更新平均奖励"""
        if len(self.buffer) == 0:
            self.stats["average_reward"] = 0.0
            return

        total_reward = sum(exp.reward for exp in self.buffer)
        self.stats["average_reward"] = total_reward / len(self.buffer)

    def _update_priority_tree(self, index: int, priority: float):
        """更新优先级树（简化实现）"""
        # 完整实现需要使用SumTree数据结构
        # 这里使用简化版本
        while len(self.priority_tree) <= index:
            self.priority_tree.append(0.0)

        self.priority_tree[index] = priority
        self.max_priority = max(self.max_priority, priority)

    def _prioritized_sampling(self, batch_size: int) -> Tuple[List[Experience], List[float]]:
        """优先采样"""
        if not self.priority_tree or len(self.priority_tree) == 0:
            return self._uniform_sampling(batch_size)

        # 计算采样概率
        priorities = self.priority_tree[:len(self.buffer)]
        total_priority = sum(priorities)

        if total_priority == 0:
            return self._uniform_sampling(batch_size)

        probabilities = [p / total_priority for p in priorities]

        # 采样
        indices = random.choices(range(len(self.buffer)), weights=probabilities, k=batch_size)
        experiences = [self.buffer[i] for i in indices]

        # 计算重要性采样权重
        weights = [
            (len(self.buffer) * prob) ** (-self.config.priority_beta)
            for prob in [probabilities[i] for i in indices]
        ]

        # 归一化权重
        max_weight = max(weights) if weights else 1.0
        weights = [w / max_weight for w in weights]

        return experiences, weights

    def _uniform_sampling(self, batch_size: int) -> Tuple[List[Experience], List[float]]:
        """均匀采样"""
        indices = random.sample(range(len(self.buffer)), batch_size)
        experiences = [self.buffer[i] for i in indices]
        weights = [1.0] * batch_size  # 均匀采样权重为1

        return experiences, weights

    def _calculate_reward_distribution(self) -> Dict[str, int]:
        """计算奖励分布"""
        distribution = {
            "very_positive": 0,  # > 0.5
            "positive": 0,       # 0.1 - 0.5
            "neutral": 0,        # -0.1 - 0.1
            "negative": 0,       # -0.5 - -0.1
            "very_negative": 0   # < -0.5
        }

        for exp in self.buffer:
            if exp.reward > 0.5:
                distribution["very_positive"] += 1
            elif exp.reward > 0.1:
                distribution["positive"] += 1
            elif exp.reward >= -0.1:
                distribution["neutral"] += 1
            elif exp.reward >= -0.5:
                distribution["negative"] += 1
            else:
                distribution["very_negative"] += 1

        return distribution

    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """分析时间模式"""
        if len(self.buffer) < 10:
            return {}

        # 按时间排序
        sorted_exps = sorted(self.buffer, key=lambda x: x.timestamp)

        # 计算最近趋势
        recent = sorted_exps[-20:]
        recent_avg_reward = sum(e.reward for e in recent) / len(recent)

        return {
            "recent_trend": "improving" if recent_avg_reward > 0 else "declining",
            "recent_average_reward": recent_avg_reward,
            "oldest_experience": sorted_exps[0].timestamp.isoformat(),
            "newest_experience": sorted_exps[-1].timestamp.isoformat()
        }

    def _find_high_value_experiences(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """查找高价值经验"""
        # 按奖励排序
        sorted_exps = sorted(self.buffer, key=lambda x: x.reward, reverse=True)

        high_value = []
        for exp in sorted_exps[:top_n]:
            high_value.append({
                "experience_id": exp.experience_id,
                "reward": exp.reward,
                "type": exp.experience_type.value,
                "pattern": exp.action.get('primary_action', 'unknown'),
                "timestamp": exp.timestamp.isoformat()
            })

        return high_value

    def _generate_learning_recommendations(self) -> List[str]:
        """生成学习建议"""
        recommendations = []

        avg_reward = self.stats["average_reward"]

        if avg_reward > 0.3:
            recommendations.append("当前策略表现良好，继续保持")
        elif avg_reward < -0.2:
            recommendations.append("需要调整策略，当前表现不佳")

        failure_count = self.stats["experiences_by_type"].get("failure", 0)
        if failure_count > len(self.buffer) * 0.3:
            recommendations.append("失败经验较多，建议分析失败原因")

        if len(self.buffer) < self.config.min_experiences:
            recommendations.append("经验数量不足，继续积累交互数据")

        return recommendations

    def _extract_common_patterns(self, experiences: List[Experience]) -> List[str]:
        """提取常见模式"""
        pattern_counts = defaultdict(int)

        for exp in experiences:
            action_type = exp.action.get('primary_action', 'unknown')
            pattern_counts[action_type] += 1

        # 返回最常见的模式
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        return [pattern for pattern, count in sorted_patterns[:5]]

    def _identify_improvement_areas(self, experiences: List[Experience]) -> List[str]:
        """识别改进领域"""
        areas = []

        # 检查失败经验
        failures = [e for e in experiences if e.reward < 0]
        if failures:
            failure_actions = set(e.action.get('primary_action', '') for e in failures)
            areas.append(f"以下行动导致失败: {', '.join(failure_actions)}")

        # 检查低置信度
        low_confidence = [
            e for e in experiences
            if e.state.get('cognitive_state', {}).get('confidence_score', 1.0) < 0.4
        ]
        if low_confidence:
            areas.append("存在低置信度决策，需要提升评估能力")

        return areas

    def _rebuild_index(self):
        """重建经验索引"""
        self.experience_index.clear()
        for i, exp in enumerate(self.buffer):
            self.experience_index[exp.experience_id] = i

    def _save_to_disk(self):
        """保存到磁盘"""
        try:
            filepath = os.path.join(
                self.config.storage_path,
                f"experience_buffer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            data = {
                "experiences": [exp.to_dict() for exp in self.buffer],
                "stats": dict(self.stats),
                "timestamp": datetime.now().isoformat()
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.logger.debug(f"💾 经验缓冲区已保存: {filepath}")

            # 清理旧文件（保留最近5个）
            self._cleanup_old_files()

        except Exception as e:
            self.logger.error(f"❌ 保存经验缓冲区失败: {str(e)}")

    def _load_from_disk(self):
        """从磁盘加载"""
        try:
            if not os.path.exists(self.config.storage_path):
                return

            # 找到最新的文件
            files = [
                f for f in os.listdir(self.config.storage_path)
                if f.startswith("experience_buffer_") and f.endswith(".json")
            ]

            if not files:
                return

            latest_file = sorted(files)[-1]
            filepath = os.path.join(self.config.storage_path, latest_file)

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 恢复经验
            for exp_data in data.get("experiences", []):
                experience = Experience(
                    experience_id=exp_data["experience_id"],
                    state=exp_data["state"],
                    action=exp_data["action"],
                    reward=exp_data["reward"],
                    next_state=exp_data["next_state"],
                    done=exp_data["done"],
                    experience_type=ExperienceType(exp_data.get("experience_type", "neutral")),
                    priority=PriorityLevel(exp_data.get("priority", "medium")),
                    timestamp=datetime.fromisoformat(exp_data["timestamp"]),
                    session_id=exp_data.get("session_id", ""),
                    tags=exp_data.get("tags", [])
                )

                self.buffer.append(experience)
                self.experience_index[experience.experience_id] = len(self.buffer) - 1

            # 恢复统计
            if "stats" in data:
                self.stats.update(data["stats"])

            self.logger.info(f"📂 已加载 {len(self.buffer)} 条历史经验")

        except Exception as e:
            self.logger.error(f"❌ 加载经验缓冲区失败: {str(e)}")

    def _cleanup_old_files(self, keep_count: int = 5):
        """清理旧文件"""
        try:
            files = [
                f for f in os.listdir(self.config.storage_path)
                if f.startswith("experience_buffer_") and f.endswith(".json")
            ]

            if len(files) > keep_count:
                sorted_files = sorted(files)
                for old_file in sorted_files[:-keep_count]:
                    filepath = os.path.join(self.config.storage_path, old_file)
                    os.remove(filepath)
                    self.logger.debug(f"🗑️ 已删除旧文件: {old_file}")

        except Exception as e:
            self.logger.error(f"❌ 清理旧文件失败: {str(e)}")
