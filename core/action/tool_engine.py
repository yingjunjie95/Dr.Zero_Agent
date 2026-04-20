"""
工具引擎 - Tool Engine
负责工具的调用执行、权限管理、性能监控和优化
提供统一的工具访问接口和安全控制机制
"""
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict


@dataclass
class ToolInfo:
    """工具信息"""
    name: str
    function: Callable
    description: str
    requires_permission: bool = False
    max_calls_per_session: Optional[int] = None
    sandbox_enabled: bool = False
    risk_level: str = "low"
    category: str = "general"
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "requires_permission": self.requires_permission,
            "max_calls_per_session": self.max_calls_per_session,
            "sandbox_enabled": self.sandbox_enabled,
            "risk_level": self.risk_level,
            "category": self.category,
            "parameters": self.parameters
        }


class ToolRegistry:
    """
    工具注册表 - 管理所有可用工具

    功能：
    - 工具注册和注销
    - 工具信息查询
    - 工具函数访问
    - 工具分类管理
    """

    def __init__(self):
        """初始化工具注册表"""
        self.tools: Dict[str, ToolInfo] = {}
        self.logger = logging.getLogger("ToolRegistry")

        self.logger.info("✅ 工具注册表初始化完成")

    def register_tool(self,
                     name: str,
                     function: Callable,
                     description: str,
                     requires_permission: bool = False,
                     max_calls_per_session: Optional[int] = None,
                     sandbox_enabled: bool = False,
                     risk_level: str = "low",
                     category: str = "general",
                     parameters: Optional[Dict[str, Any]] = None):
        """
        注册工具

        Args:
            name: 工具名称
            function: 工具函数
            description: 工具描述
            requires_permission: 是否需要权限
            max_calls_per_session: 会话最大调用次数
            sandbox_enabled: 是否启用沙箱
            risk_level: 风险等级
            category: 工具分类
            parameters: 参数说明
        """
        if name in self.tools:
            self.logger.warning(f"⚠️ 工具已存在，将被覆盖: {name}")

        tool_info = ToolInfo(
            name=name,
            function=function,
            description=description,
            requires_permission=requires_permission,
            max_calls_per_session=max_calls_per_session,
            sandbox_enabled=sandbox_enabled,
            risk_level=risk_level,
            category=category,
            parameters=parameters or {}
        )

        self.tools[name] = tool_info

        self.logger.info(f"🔧 工具已注册: {name} (类别: {category}, 风险: {risk_level})")

    def unregister_tool(self, name: str) -> bool:
        """
        注销工具

        Args:
            name: 工具名称

        Returns:
            是否成功注销
        """
        if name in self.tools:
            del self.tools[name]
            self.logger.info(f"🗑️ 工具已注销: {name}")
            return True

        self.logger.warning(f"⚠️ 工具不存在: {name}")
        return False

    def get_tool_function(self, name: str) -> Optional[Callable]:
        """
        获取工具函数

        Args:
            name: 工具名称

        Returns:
            工具函数，不存在返回None
        """
        if name in self.tools:
            return self.tools[name].function

        self.logger.warning(f"⚠️ 尝试获取不存在的工具: {name}")
        return None

    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取工具信息

        Args:
            name: 工具名称

        Returns:
            工具信息字典，不存在返回None
        """
        if name in self.tools:
            return self.tools[name].to_dict()

        return None

    def get_tool_names(self) -> List[str]:
        """获取所有工具名称"""
        return list(self.tools.keys())

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """获取所有可用工具信息"""
        return [tool.to_dict() for tool in self.tools.values()]

    def get_tools_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        按分类获取工具

        Args:
            category: 工具分类

        Returns:
            工具信息列表
        """
        return [
            tool.to_dict()
            for tool in self.tools.values()
            if tool.category == category
        ]

    def get_tools_requiring_permission(self) -> List[str]:
        """获取需要权限的工具列表"""
        return [
            name for name, tool in self.tools.items()
            if tool.requires_permission
        ]

    def is_tool_available(self, name: str) -> bool:
        """
        检查工具是否可用

        Args:
            name: 工具名称

        Returns:
            是否可用
        """
        return name in self.tools

    def clear_all(self):
        """清空所有工具"""
        self.tools.clear()
        self.logger.info("🧹 工具注册表已清空")

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        category_stats = {}
        for tool in self.tools.values():
            cat = tool.category
            category_stats[cat] = category_stats.get(cat, 0) + 1

        return {
            "total_tools": len(self.tools),
            "tools_requiring_permission": sum(
                1 for tool in self.tools.values()
                if tool.requires_permission
            ),
            "category_distribution": category_stats,
            "tool_names": list(self.tools.keys())
        }


class ToolEngine:
    """
    工具引擎 - 负责工具的执行和管理

    功能：
    - 工具调用执行
    - 权限验证和控制
    - 调用频率限制
    - 性能监控和统计
    - 工具使用优化
    - 错误处理和恢复
    """

    def __init__(self, tool_registry: ToolRegistry):
        """
        初始化工具引擎

        Args:
            tool_registry: 工具注册表实例
        """
        self.tool_registry = tool_registry
        self.logger = logging.getLogger("ToolEngine")

        # 调用统计
        self.call_statistics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0,
            "last_called": None
        })

        # 会话调用计数
        self.session_call_counts: Dict[str, int] = defaultdict(int)

        # 性能历史
        self.performance_history: List[Dict[str, Any]] = []
        self.max_history_size = 100

        self.logger.info("✅ 工具引擎初始化完成")

    def execute_tool(
        self,
        tool_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        执行工具调用

        Args:
            tool_name: 工具名称
            parameters: 工具参数
            context: 执行上下文

        Returns:
            执行结果字典
        """
        start_time = time.time()
        session_id = context.get('session_id', 'default') if context else 'default'

        try:
            self.logger.debug(f"🔧 执行工具: {tool_name}")

            # 1. 验证工具是否存在
            if not self.tool_registry.is_tool_available(tool_name):
                error_msg = f"工具不存在: {tool_name}"
                self.logger.error(f"❌ {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "tool_name": tool_name,
                    "execution_time": 0.0
                }

            # 2. 检查调用频率限制
            if not self._check_call_limit(tool_name, session_id):
                error_msg = f"工具调用次数已达上限: {tool_name}"
                self.logger.warning(f"⚠️ {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "tool_name": tool_name,
                    "execution_time": 0.0
                }

            # 3. 获取工具函数
            tool_function = self.tool_registry.get_tool_function(tool_name)
            if tool_function is None:
                error_msg = f"无法获取工具函数: {tool_name}"
                self.logger.error(f"❌ {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "tool_name": tool_name,
                    "execution_time": 0.0
                }

            # 4. 执行工具
            parameters = parameters or {}
            result = tool_function(**parameters)

            # 5. 计算执行时间
            execution_time = time.time() - start_time

            # 6. 更新统计
            self._update_statistics(tool_name, True, execution_time, session_id)

            self.logger.debug(
                f"✅ 工具执行成功: {tool_name} - "
                f"耗时: {execution_time:.2f}s"
            )

            return {
                "success": True,
                "output": result,
                "tool_name": tool_name,
                "execution_time": execution_time,
                "parameters_used": parameters
            }

        except Exception as e:
            execution_time = time.time() - start_time

            # 更新失败统计
            self._update_statistics(tool_name, False, execution_time, session_id)

            error_msg = f"工具执行失败: {str(e)}"
            self.logger.error(f"❌ {error_msg}", exc_info=True)

            return {
                "success": False,
                "error": error_msg,
                "exception_type": type(e).__name__,
                "tool_name": tool_name,
                "execution_time": execution_time,
                "parameters_used": parameters or {}
            }

    def _check_call_limit(self, tool_name: str, session_id: str) -> bool:
        """
        检查调用频率限制

        Args:
            tool_name: 工具名称
            session_id: 会话ID

        Returns:
            是否允许调用
        """
        tool_info = self.tool_registry.get_tool_info(tool_name)
        if not tool_info:
            return False

        max_calls = tool_info.get('max_calls_per_session')
        if max_calls is None:
            return True

        current_count = self.session_call_counts.get(f"{session_id}:{tool_name}", 0)
        return current_count < max_calls

    def _update_statistics(
        self,
        tool_name: str,
        success: bool,
        execution_time: float,
        session_id: str
    ):
        """
        更新工具调用统计

        Args:
            tool_name: 工具名称
            success: 是否成功
            execution_time: 执行时间
            session_id: 会话ID
        """
        stats = self.call_statistics[tool_name]
        stats["total_calls"] += 1

        if success:
            stats["successful_calls"] += 1
        else:
            stats["failed_calls"] += 1

        # 更新平均执行时间
        stats["total_execution_time"] += execution_time
        stats["average_execution_time"] = (
            stats["total_execution_time"] / stats["total_calls"]
        )

        stats["last_called"] = datetime.now().isoformat()

        # 更新会话调用计数
        session_key = f"{session_id}:{tool_name}"
        self.session_call_counts[session_key] = (
            self.session_call_counts.get(session_key, 0) + 1
        )

        # 记录性能历史
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "success": success,
            "execution_time": execution_time,
            "session_id": session_id
        })

        # 限制历史记录大小
        if len(self.performance_history) > self.max_history_size:
            self.performance_history.pop(0)

    def optimize_tool_usage(self, interaction_data: Dict[str, Any]):
        """
        优化工具使用策略

        Args:
            interaction_data: 交互数据
        """
        outcome = interaction_data.get('outcome', 'unknown')
        decision = interaction_data.get('decision', {})
        execution_metrics = interaction_data.get('execution_metrics', {})

        # 分析工具使用效果
        if execution_metrics.get('tool_calls', 0) > 0:
            success = outcome == 'success'

            # 这里可以添加工具使用模式的分析和优化逻辑
            # 例如：识别低效的工具组合、调整调用顺序等
            self.logger.debug(
                f"📊 工具使用优化分析 - "
                f"结果: {outcome}, "
                f"工具调用数: {execution_metrics['tool_calls']}"
            )

    def get_tool_statistics(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取工具统计信息

        Args:
            tool_name: 工具名称，None表示获取所有工具统计

        Returns:
            统计信息字典
        """
        if tool_name:
            if tool_name in self.call_statistics:
                return {
                    tool_name: dict(self.call_statistics[tool_name])
                }
            else:
                return {}

        return {
            name: dict(stats)
            for name, stats in self.call_statistics.items()
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.call_statistics:
            return {"message": "无工具调用记录"}

        total_calls = sum(
            stats["total_calls"]
            for stats in self.call_statistics.values()
        )
        total_success = sum(
            stats["successful_calls"]
            for stats in self.call_statistics.values()
        )
        total_failures = sum(
            stats["failed_calls"]
            for stats in self.call_statistics.values()
        )

        overall_success_rate = (
            total_success / total_calls if total_calls > 0 else 0.0
        )

        # 找出最常用和最慢的工具
        most_used_tool = max(
            self.call_statistics.items(),
            key=lambda x: x[1]["total_calls"],
            default=(None, None)
        )

        slowest_tool = max(
            self.call_statistics.items(),
            key=lambda x: x[1]["average_execution_time"],
            default=(None, None)
        )

        return {
            "total_calls": total_calls,
            "total_success": total_success,
            "total_failures": total_failures,
            "overall_success_rate": round(overall_success_rate, 3),
            "unique_tools_used": len(self.call_statistics),
            "most_used_tool": {
                "name": most_used_tool[0],
                "calls": most_used_tool[1]["total_calls"]
            } if most_used_tool[0] else None,
            "slowest_tool": {
                "name": slowest_tool[0],
                "avg_time": round(slowest_tool[1]["average_execution_time"], 3)
            } if slowest_tool[0] else None,
            "recent_performance": self.performance_history[-10:]
        }

    def reset_session_counts(self, session_id: str):
        """
        重置会话调用计数

        Args:
            session_id: 会话ID
        """
        keys_to_remove = [
            key for key in self.session_call_counts.keys()
            if key.startswith(f"{session_id}:")
        ]

        for key in keys_to_remove:
            del self.session_call_counts[key]

        self.logger.debug(f"🔄 已重置会话 {session_id} 的调用计数")

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """获取可用工具列表"""
        return self.tool_registry.get_available_tools()
