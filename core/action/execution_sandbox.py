"""
执行沙箱 - Execution Sandbox
为高风险工具提供安全的隔离执行环境
实现代码执行、资源限制和安全防护
专为CPU环境优化，确保系统稳定性
"""
import logging
import time
import sys
import os
import io
import re
import ast
import signal
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr


class SandboxStatus(Enum):
    """沙箱状态"""
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"
    SECURITY_VIOLATION = "security_violation"


class SecurityLevel(Enum):
    """安全等级"""
    MINIMAL = "minimal"  # 最小限制（仅基础检查）
    STANDARD = "standard"  # 标准限制（推荐）
    STRICT = "strict"  # 严格限制（高风险操作）
    MAXIMUM = "maximum"  # 最高限制（未知代码）


@dataclass
class SandboxConfig:
    """沙箱配置"""
    max_execution_time: float = 5.0  # 最大执行时间（秒）
    max_memory_mb: int = 256  # 最大内存使用（MB）
    max_output_size: int = 10000  # 最大输出大小（字符）
    allowed_modules: List[str] = field(default_factory=list)  # 允许的模块
    forbidden_modules: List[str] = field(default_factory=list)  # 禁止的模块
    allow_file_access: bool = False  # 是否允许文件访问
    allow_network_access: bool = False  # 是否允许网络访问
    security_level: SecurityLevel = SecurityLevel.STANDARD

    def __post_init__(self):
        """初始化后设置默认值"""
        if not self.allowed_modules:
            self.allowed_modules = self._get_default_allowed_modules()
        if not self.forbidden_modules:
            self.forbidden_modules = self._get_default_forbidden_modules()

    @staticmethod
    def _get_default_allowed_modules() -> List[str]:
        """获取默认允许的模块列表"""
        return [
            'math', 'random', 'datetime', 'collections',
            'itertools', 'functools', 're', 'string',
            'json', 'csv', 'hashlib', 'base64',
            'typing', 'dataclasses', 'enum'
        ]

    @staticmethod
    def _get_default_forbidden_modules() -> List[str]:
        """获取默认禁止的模块列表"""
        return [
            'os', 'sys', 'subprocess', 'multiprocessing',
            'threading', 'socket', 'http', 'urllib',
            'requests', 'ftplib', 'smtplib', 'pickle',
            'shelve', 'ctypes', 'importlib', 'pkgutil',
            'inspect', 'traceback', 'pdb', 'profile',
            'cProfile', 'resource', 'fcntl', 'posix',
            'pwd', 'grp', 'signal', 'mmap'
        ]


@dataclass
class SandboxResult:
    """沙箱执行结果"""
    success: bool
    output: str
    error: str
    status: SandboxStatus
    execution_time: float
    memory_used_mb: float
    exit_code: int

    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "output": self.output[:1000],  # 限制输出长度
            "error": self.error[:500],
            "status": self.status.value,
            "execution_time": round(self.execution_time, 3),
            "memory_used_mb": round(self.memory_used_mb, 2),
            "exit_code": self.exit_code,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class CodeSecurityChecker:
    """代码安全检查器"""

    def __init__(self, config: SandboxConfig):
        self.config = config
        self.logger = logging.getLogger("CodeSecurityChecker")

        # 危险的AST节点类型
        self.dangerous_nodes = {
            ast.Import: "import语句",
            ast.ImportFrom: "from...import语句",
            ast.Call: "函数调用",
        }

    def check_code_safety(self, code: str) -> Tuple[bool, List[str]]:
        """
        检查代码安全性

        Args:
            code: 待检查的代码

        Returns:
            (是否安全, 安全问题列表)
        """
        issues = []

        try:
            # 1. 解析AST
            tree = ast.parse(code)

            # 2. 检查导入语句
            import_issues = self._check_imports(tree)
            issues.extend(import_issues)

            # 3. 检查危险函数调用
            call_issues = self._check_dangerous_calls(tree)
            issues.extend(call_issues)

            # 4. 检查字符串中的危险模式
            pattern_issues = self._check_dangerous_patterns(code)
            issues.extend(pattern_issues)

            # 5. 检查代码复杂度
            complexity_issues = self._check_complexity(tree)
            issues.extend(complexity_issues)

            is_safe = len(issues) == 0
            return is_safe, issues

        except SyntaxError as e:
            return False, [f"语法错误: {str(e)}"]
        except Exception as e:
            return False, [f"安全检查失败: {str(e)}"]

    def _check_imports(self, tree: ast.AST) -> List[str]:
        """检查导入语句"""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if module_name in self.config.forbidden_modules:
                        issues.append(f"禁止导入模块: {module_name}")
                    elif self.config.allowed_modules and module_name not in self.config.allowed_modules:
                        if self.config.security_level in [SecurityLevel.STRICT, SecurityLevel.MAXIMUM]:
                            issues.append(f"未授权模块: {module_name}")

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if module_name in self.config.forbidden_modules:
                        issues.append(f"禁止导入模块: {module_name}")

        return issues

    def _check_dangerous_calls(self, tree: ast.AST) -> List[str]:
        """检查危险的函数调用"""
        issues = []
        dangerous_functions = [
            'eval', 'exec', 'compile', '__import__',
            'globals', 'locals', 'vars', 'dir',
            'getattr', 'setattr', 'delattr',
            'open', 'input', 'breakpoint'
        ]

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = None

                # 直接函数调用
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                # 属性调用
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in dangerous_functions:
                    issues.append(f"危险函数调用: {func_name}")

        return issues

    def _check_dangerous_patterns(self, code: str) -> List[str]:
        """检查危险的模式"""
        issues = []

        dangerous_patterns = [
            (r'__\w+__', "双下划线魔术方法"),
            (r'os\.(system|popen|exec)', "系统命令执行"),
            (r'subprocess\.', "子进程创建"),
            (r'eval\s*\(', "eval函数"),
            (r'exec\s*\(', "exec函数"),
        ]

        for pattern, description in dangerous_patterns:
            if re.search(pattern, code):
                issues.append(f"检测到危险模式: {description}")

        return issues

    def _check_complexity(self, tree: ast.AST) -> List[str]:
        """检查代码复杂度"""
        issues = []

        # 计算嵌套深度
        max_depth = self._calculate_max_depth(tree)
        if max_depth > 5:
            issues.append(f"嵌套深度过大: {max_depth}层")

        # 计算循环数量
        loop_count = sum(1 for node in ast.walk(tree)
                        if isinstance(node, (ast.For, ast.While)))
        if loop_count > 3:
            issues.append(f"循环数量过多: {loop_count}个")

        return issues

    @staticmethod
    def _calculate_max_depth(node: ast.AST, current_depth: int = 0) -> int:
        """计算AST最大嵌套深度"""
        max_depth = current_depth

        for child in ast.iter_child_nodes(node):
            child_depth = CodeSecurityChecker._calculate_max_depth(
                child, current_depth + 1
            )
            max_depth = max(max_depth, child_depth)

        return max_depth


class ExecutionSandbox:
    """
    执行沙箱

    功能：
    - 代码安全检查和过滤
    - 资源限制（时间、内存）
    - 隔离执行环境
    - 输出捕获和限制
    - 异常处理和恢复
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        """
        初始化执行沙箱

        Args:
            config: 沙箱配置
        """
        self.config = config or SandboxConfig()
        self.logger = logging.getLogger("ExecutionSandbox")

        # 安全检查器
        self.security_checker = CodeSecurityChecker(self.config)

        # 执行统计
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "security_violations": 0,
            "timeouts": 0,
            "avg_execution_time": 0.0
        }

        # 执行历史
        self.execution_history: List[SandboxResult] = []
        self.max_history_size = 50

        self.logger.info(f"✅ 执行沙箱初始化完成 - 安全等级: {self.config.security_level.value}")

    def execute_code(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> SandboxResult:
        """
        在沙箱中执行代码

        Args:
            code: 要执行的Python代码
            context: 执行上下文（变量字典）
            timeout: 超时时间（秒），覆盖配置值

        Returns:
            沙箱执行结果
        """
        start_time = time.time()
        self.stats['total_executions'] += 1

        try:
            self.logger.debug("🔒 开始沙箱执行...")

            # 1. 安全检查
            is_safe, issues = self.security_checker.check_code_safety(code)
            if not is_safe:
                self.stats['security_violations'] += 1
                self.logger.warning(f"⚠️ 安全检查失败: {issues}")

                result = SandboxResult(
                    success=False,
                    output="",
                    error=f"安全检查失败: {'; '.join(issues)}",
                    status=SandboxStatus.SECURITY_VIOLATION,
                    execution_time=time.time() - start_time,
                    memory_used_mb=0.0,
                    exit_code=-1,
                    metadata={"security_issues": issues}
                )

                self._record_execution(result)
                return result

            # 2. 准备执行环境
            exec_timeout = timeout or self.config.max_execution_time
            execution_context = self._prepare_context(context)

            # 3. 执行代码
            self.logger.debug("▶️ 执行代码...")
            output_buffer = io.StringIO()
            error_buffer = io.StringIO()

            try:
                # 重定向标准输出和错误
                with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                    # 执行代码
                    exec(
                        compile(code, '<sandbox>', 'exec'),
                        execution_context
                    )

                output = output_buffer.getvalue()
                error = error_buffer.getvalue()

                # 限制输出大小
                if len(output) > self.config.max_output_size:
                    output = output[:self.config.max_output_size] + "\n... [输出被截断]"

                execution_time = time.time() - start_time

                # 检查超时
                if execution_time > exec_timeout:
                    self.stats['timeouts'] += 1
                    result = SandboxResult(
                        success=False,
                        output=output,
                        error=f"执行超时: {execution_time:.2f}s > {exec_timeout}s",
                        status=SandboxStatus.TIMEOUT,
                        execution_time=execution_time,
                        memory_used_mb=self._estimate_memory_usage(execution_context),
                        exit_code=-2,
                        metadata={"timeout": exec_timeout}
                    )
                else:
                    # 执行成功
                    self.stats['successful_executions'] += 1
                    result = SandboxResult(
                        success=True,
                        output=output,
                        error=error,
                        status=SandboxStatus.COMPLETED,
                        execution_time=execution_time,
                        memory_used_mb=self._estimate_memory_usage(execution_context),
                        exit_code=0,
                        metadata={"context_keys": list(execution_context.keys())}
                    )

            except Exception as e:
                execution_time = time.time() - start_time
                self.stats['failed_executions'] += 1

                error_msg = f"{type(e).__name__}: {str(e)}"
                self.logger.error(f"❌ 执行错误: {error_msg}")

                result = SandboxResult(
                    success=False,
                    output=output_buffer.getvalue(),
                    error=error_msg,
                    status=SandboxStatus.ERROR,
                    execution_time=execution_time,
                    memory_used_mb=self._estimate_memory_usage(execution_context),
                    exit_code=-3,
                    metadata={"exception_type": type(e).__name__}
                )

            # 记录执行
            self._record_execution(result)

            self.logger.debug(
                f"✨ 沙箱执行完成 - 状态: {result.status.value}, "
                f"耗时: {result.execution_time:.3f}s"
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ 沙箱执行异常: {str(e)}", exc_info=True)

            result = SandboxResult(
                success=False,
                output="",
                error=f"沙箱异常: {str(e)}",
                status=SandboxStatus.ERROR,
                execution_time=execution_time,
                memory_used_mb=0.0,
                exit_code=-4
            )

            self._record_execution(result)
            return result

    def execute_function(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> SandboxResult:
        """
        在沙箱中执行函数

        Args:
            func: 要执行的函数
            args: 位置参数
            kwargs: 关键字参数
            timeout: 超时时间

        Returns:
            沙箱执行结果
        """
        kwargs = kwargs or {}

        # 将函数调用转换为代码
        func_name = getattr(func, '__name__', 'unknown')
        code_lines = [
            "import inspect",
            f"result = func(*args, **kwargs)",
            "print(result)"
        ]
        code = "\n".join(code_lines)

        context = {
            'func': func,
            'args': args,
            'kwargs': kwargs
        }

        return self.execute_code(code, context, timeout)

    def _prepare_context(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """准备执行上下文"""
        safe_builtins = {
            'print': print,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sorted': sorted,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'int': int,
            'float': float,
            'str': str,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'bool': bool,
            'type': type,
            'isinstance': isinstance,
            'issubclass': issubclass,
        }

        execution_context = {'__builtins__': safe_builtins}

        if context:
            execution_context.update(context)

        return execution_context

    def _estimate_memory_usage(self, context: Dict[str, Any]) -> float:
        """估算内存使用量（MB）"""
        try:
            total_size = 0
            for key, value in context.items():
                if key.startswith('__'):
                    continue
                try:
                    size = sys.getsizeof(value)
                    total_size += size
                except:
                    pass

            return total_size / (1024 * 1024)  # 转换为MB
        except:
            return 0.0

    def _record_execution(self, result: SandboxResult):
        """记录执行结果"""
        self.execution_history.append(result)

        # 限制历史记录大小
        if len(self.execution_history) > self.max_history_size:
            self.execution_history.pop(0)

        # 更新平均执行时间
        total_time = sum(r.execution_time for r in self.execution_history)
        self.stats['avg_execution_time'] = total_time / len(self.execution_history)

    def get_statistics(self) -> Dict[str, Any]:
        """获取执行统计"""
        return {
            **self.stats,
            "history_size": len(self.execution_history),
            "success_rate": (
                self.stats['successful_executions'] / self.stats['total_executions']
                if self.stats['total_executions'] > 0 else 0.0
            ),
            "config": {
                "max_execution_time": self.config.max_execution_time,
                "max_memory_mb": self.config.max_memory_mb,
                "security_level": self.config.security_level.value,
                "allowed_modules_count": len(self.config.allowed_modules),
                "forbidden_modules_count": len(self.config.forbidden_modules)
            }
        }

    def get_recent_executions(self, count: int = 10) -> List[Dict[str, Any]]:
        """获取最近的执行记录"""
        recent = self.execution_history[-count:]
        return [r.to_dict() for r in reversed(recent)]

    def clear_history(self):
        """清空执行历史"""
        self.execution_history.clear()
        self.logger.info("🧹 执行历史已清空")
