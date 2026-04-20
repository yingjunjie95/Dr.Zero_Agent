"""
工具注册表 - Tool Registry
管理和注册所有可用工具
提供工具的元数据、权限控制和访问接口
专为CPU环境优化，轻量级且高效的工具管理系统
"""
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ToolInfo:
    """工具信息"""
    name: str
    function: Callable
    description: str
    requires_permission: bool = False
    max_calls_per_session: Optional[int] = None
    category: str = "general"
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "requires_permission": self.requires_permission,
            "max_calls_per_session": self.max_calls_per_session,
            "category": self.category,
            "enabled": self.enabled
        }


class ToolRegistry:
    """
    工具注册表

    功能：
    - 工具注册：注册和管理所有可用工具
    - 元数据管理：维护工具的详细描述和配置
    - 权限控制：标记需要用户确认的工具
    - 分类管理：按类别组织工具
    - 查询接口：提供工具查找和列表功能
    """

    def __init__(self):
        """初始化工具注册表"""
        self.tools: Dict[str, ToolInfo] = {}
        self.logger = logging.getLogger("ToolRegistry")

        self.logger.info("✅ 工具注册表初始化完成")

    def register_tool(
        self,
        name: str,
        function: Callable,
        description: str,
        requires_permission: bool = False,
        max_calls_per_session: Optional[int] = None,
        category: str = "general"
    ):
        """
        注册工具

        Args:
            name: 工具名称
            function: 工具函数
            description: 工具描述
            requires_permission: 是否需要用户权限
            max_calls_per_session: 每次会话最大调用次数
            category: 工具类别
        """
        if name in self.tools:
            self.logger.warning(f"⚠️ 工具 {name} 已存在，将被覆盖")

        tool_info = ToolInfo(
            name=name,
            function=function,
            description=description,
            requires_permission=requires_permission,
            max_calls_per_session=max_calls_per_session,
            category=category
        )

        self.tools[name] = tool_info

        self.logger.info(
            f"📝 工具已注册: {name} (类别: {category}, "
            f"需要权限: {requires_permission})"
        )

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
        else:
            self.logger.warning(f"⚠️ 工具不存在: {name}")
            return False

    def get_tool(self, name: str) -> Optional[ToolInfo]:
        """
        获取工具信息

        Args:
            name: 工具名称

        Returns:
            工具信息，不存在则返回None
        """
        return self.tools.get(name)

    def get_tool_function(self, name: str) -> Optional[Callable]:
        """
        获取工具函数

        Args:
            name: 工具名称

        Returns:
            工具函数，不存在则返回None
        """
        tool = self.tools.get(name)
        return tool.function if tool else None

    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取工具信息字典

        Args:
            name: 工具名称

        Returns:
            工具信息字典，不存在则返回None
        """
        tool = self.tools.get(name)
        return tool.to_dict() if tool else None

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        获取所有可用工具列表

        Returns:
            工具信息列表
        """
        return [
            tool.to_dict()
            for tool in self.tools.values()
            if tool.enabled
        ]

    def get_tool_names(self) -> List[str]:
        """
        获取所有工具名称

        Returns:
            工具名称列表
        """
        return list(self.tools.keys())

    def get_tools_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        按类别获取工具

        Args:
            category: 工具类别

        Returns:
            该类别下的工具列表
        """
        return [
            tool.to_dict()
            for tool in self.tools.values()
            if tool.category == category and tool.enabled
        ]

    def get_categories(self) -> List[str]:
        """
        获取所有工具类别

        Returns:
            类别列表
        """
        categories = set(tool.category for tool in self.tools.values())
        return sorted(list(categories))

    def enable_tool(self, name: str) -> bool:
        """
        启用工具

        Args:
            name: 工具名称

        Returns:
            是否成功启用
        """
        tool = self.tools.get(name)
        if tool:
            tool.enabled = True
            self.logger.info(f"✅ 工具已启用: {name}")
            return True
        return False

    def disable_tool(self, name: str) -> bool:
        """
        禁用工具

        Args:
            name: 工具名称

        Returns:
            是否成功禁用
        """
        tool = self.tools.get(name)
        if tool:
            tool.enabled = False
            self.logger.info(f"⏸️ 工具已禁用: {name}")
            return True
        return False

    def tool_exists(self, name: str) -> bool:
        """
        检查工具是否存在

        Args:
            name: 工具名称

        Returns:
            是否存在
        """
        return name in self.tools

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取注册表统计信息

        Returns:
            统计信息字典
        """
        category_stats = {}
        for tool in self.tools.values():
            if tool.category not in category_stats:
                category_stats[tool.category] = 0
            category_stats[tool.category] += 1

        return {
            "total_tools": len(self.tools),
            "enabled_tools": sum(1 for t in self.tools.values() if t.enabled),
            "disabled_tools": sum(1 for t in self.tools.values() if not t.enabled),
            "tools_requiring_permission": sum(
                1 for t in self.tools.values() if t.requires_permission
            ),
            "categories": category_stats,
            "tool_names": list(self.tools.keys())
        }

    def clear_all(self):
        """清空所有注册的工具"""
        self.tools.clear()
        self.logger.info("🧹 所有工具已清空")