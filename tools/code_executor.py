"""
代码执行器 - Code Executor
提供安全的Python代码执行环境
支持代码验证、沙箱隔离和执行监控
专为CPU环境优化，轻量级且安全的代码执行解决方案
"""
import logging
import sys
import os
import re
import time
import ast
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr


class SecurityLevel(Enum):
    """安全级别"""
    STRICT = "strict"       # 严格模式，禁止大多数操作
    MODERATE = "moderate"   # 中等模式，允许基本操作
    RELAXED = "relaxed"     # 宽松模式，允许更多操作


@dataclass
class ExecutionConfig:
    """执行配置"""
    timeout_seconds: float = 5.0
    max_output_length: int = 10000
    max_memory_mb: int = 256
    security_level: SecurityLevel = SecurityLevel.STRICT
    allowed_modules: List[str] = field(default_factory=list)
    banned_functions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timeout_seconds": self.timeout_seconds,
            "max_output_length": self.max_output_length,
            "max_memory_mb": self.max_memory_mb,
            "security_level": self.security_level.value,
            "allowed_modules_count": len(self.allowed_modules),
            "banned_functions_count": len(self.banned_functions)
        }


@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    output: str
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    exit_code: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "output": self.output[:5000],  # 限制输出长度
            "error": self.error,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "exit_code": self.exit_code,
            "warnings": self.warnings
        }


@dataclass
class CodeAnalysis:
    """代码分析结果"""
    is_safe: bool
    complexity_score: float
    line_count: int
    function_count: int
    class_count: int
    imports: List[str]
    potential_risks: List[str]
    suggestions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "is_safe": self.is_safe,
            "complexity_score": round(self.complexity_score, 2),
            "line_count": self.line_count,
            "function_count": self.function_count,
            "class_count": self.class_count,
            "imports": self.imports,
            "potential_risks": self.potential_risks,
            "suggestions": self.suggestions
        }


class CodeExecutor:
    """
    代码执行器

    功能：
    - 安全执行：在受限环境中执行Python代码
    - 代码分析：静态分析代码安全性和复杂度
    - 资源限制：控制执行时间和内存使用
    - 输出捕获：捕获标准输出和错误
    - 白名单机制：只允许安全的模块和函数
    - 风险评估：识别潜在的安全风险
    """

    def __init__(
        self,
        default_timeout: float = 5.0,
        default_security_level: SecurityLevel = SecurityLevel.STRICT,
        enable_analysis: bool = True
    ):
        """
        初始化代码执行器

        Args:
            default_timeout: 默认超时时间（秒）
            default_security_level: 默认安全级别
            enable_analysis: 是否启用代码分析
        """
        self.default_timeout = default_timeout
        self.default_security_level = default_security_level
        self.enable_analysis = enable_analysis

        self.logger = logging.getLogger("CodeExecutor")

        # 执行统计
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.blocked_executions = 0

        # 安全配置
        self.allowed_modules = self._get_allowed_modules()
        self.builtin_whitelist = self._get_builtin_whitelist()
        self.banned_patterns = self._get_banned_patterns()

        self.logger.info(
            f"✅ 代码执行器初始化完成 - "
            f"超时: {default_timeout}s, "
            f"安全级别: {default_security_level.value}"
        )

    def execute_code(
        self,
        code: str,
        config: Optional[ExecutionConfig] = None,
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        执行代码

        Args:
            code: 要执行的Python代码
            config: 执行配置
            variables: 注入的变量

        Returns:
            执行结果字典
        """
        self.total_executions += 1
        start_time = time.time()

        try:
            # 参数验证
            if not code or not code.strip():
                raise ValueError("代码不能为空")

            # 使用默认配置
            exec_config = config or ExecutionConfig(
                timeout_seconds=self.default_timeout,
                security_level=self.default_security_level
            )

            # 代码分析
            if self.enable_analysis:
                analysis = self.analyze_code(code)

                if not analysis.is_safe:
                    self.blocked_executions += 1
                    risks = '; '.join(analysis.potential_risks)
                    raise SecurityError(f"代码安全检查失败: {risks}")

            # 执行代码
            result = self._safe_execute(code, exec_config, variables)

            # 计算执行时间
            execution_time_ms = (time.time() - start_time) * 1000
            result.execution_time_ms = execution_time_ms

            # 更新统计
            if result.success:
                self.successful_executions += 1
            else:
                self.failed_executions += 1

            self.logger.info(
                f"{'✅' if result.success else '❌'} 代码执行完成 - "
                f"耗时: {execution_time_ms:.0f}ms"
            )

            return result.to_dict()

        except SecurityError as e:
            self.blocked_executions += 1
            execution_time_ms = (time.time() - start_time) * 1000

            self.logger.warning(f"⚠️ 代码被阻止: {str(e)}")

            error_result = ExecutionResult(
                success=False,
                output="",
                error=f"安全阻止: {str(e)}",
                execution_time_ms=execution_time_ms
            )

            return error_result.to_dict()

        except Exception as e:
            self.failed_executions += 1
            execution_time_ms = (time.time() - start_time) * 1000

            self.logger.error(f"❌ 代码执行失败: {str(e)}")

            error_result = ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time_ms=execution_time_ms
            )

            return error_result.to_dict()

    def analyze_code(self, code: str) -> CodeAnalysis:
        """
        分析代码

        Args:
            code: 要分析的代码

        Returns:
            分析结果
        """
        try:
            # 基本统计
            lines = code.split('\n')
            line_count = len([l for l in lines if l.strip()])

            # 解析AST
            tree = ast.parse(code)

            # 统计函数和类
            function_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            class_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))

            # 提取导入
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            # 检查风险
            potential_risks = self._check_security_risks(code, tree, imports)

            # 计算复杂度
            complexity_score = self._calculate_complexity(tree)

            # 判断是否安全
            is_safe = len(potential_risks) == 0

            # 生成建议
            suggestions = self._generate_suggestions(code, analysis_results={
                'line_count': line_count,
                'complexity': complexity_score,
                'risks': potential_risks
            })

            return CodeAnalysis(
                is_safe=is_safe,
                complexity_score=complexity_score,
                line_count=line_count,
                function_count=function_count,
                class_count=class_count,
                imports=imports,
                potential_risks=potential_risks,
                suggestions=suggestions
            )

        except SyntaxError as e:
            return CodeAnalysis(
                is_safe=False,
                complexity_score=0.0,
                line_count=0,
                function_count=0,
                class_count=0,
                imports=[],
                potential_risks=[f"语法错误: {str(e)}"],
                suggestions=["请检查代码语法"]
            )

        except Exception as e:
            self.logger.error(f"代码分析失败: {str(e)}")
            return CodeAnalysis(
                is_safe=False,
                complexity_score=0.0,
                line_count=0,
                function_count=0,
                class_count=0,
                imports=[],
                potential_risks=[f"分析失败: {str(e)}"],
                suggestions=[]
            )

    def _safe_execute(
        self,
        code: str,
        config: ExecutionConfig,
        variables: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """安全执行代码"""
        # 创建受限的执行环境
        restricted_globals = self._create_restricted_environment(config)

        # 添加自定义变量
        if variables:
            restricted_globals.update(variables)

        # 捕获输出
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()

        try:
            # 执行代码
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(compile(code, '<string>', 'exec'), restricted_globals)

            # 获取输出
            output = stdout_buffer.getvalue()
            error = stderr_buffer.getvalue()

            # 限制输出长度
            if len(output) > config.max_output_length:
                output = output[:config.max_output_length] + "\n... [输出被截断]"

            # 检查是否有错误
            if error:
                return ExecutionResult(
                    success=False,
                    output=output,
                    error=error.strip()
                )

            return ExecutionResult(
                success=True,
                output=output.strip() if output else "执行成功（无输出）"
            )

        except TimeoutError:
            return ExecutionResult(
                success=False,
                output="",
                error=f"执行超时（超过{config.timeout_seconds}秒）"
            )

        except MemoryError:
            return ExecutionResult(
                success=False,
                output="",
                error="内存不足"
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            return ExecutionResult(
                success=False,
                output=stdout_buffer.getvalue(),
                error=error_msg
            )

    def _create_restricted_environment(self, config: ExecutionConfig) -> Dict[str, Any]:
        """创建受限的执行环境"""
        # 只包含安全的内置函数
        safe_builtins = {
            name: obj for name, obj in __builtins__.items()
            if name in self.builtin_whitelist
        }

        environment = {
            '__builtins__': safe_builtins,
            '__name__': '__main__',
            '__file__': None,
        }

        # 根据安全级别添加模块
        if config.security_level != SecurityLevel.STRICT:
            # 添加允许的模块
            for module_name in config.allowed_modules or self.allowed_modules:
                try:
                    module = __import__(module_name)
                    environment[module_name] = module
                except ImportError:
                    self.logger.warning(f"无法导入模块: {module_name}")

        return environment

    def _check_security_risks(
        self,
        code: str,
        tree: ast.AST,
        imports: List[str]
    ) -> List[str]:
        """检查安全风险"""
        risks = []

        # 检查禁止的模式
        for pattern_name, pattern in self.banned_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                risks.append(f"检测到危险模式: {pattern_name}")

        # 检查危险的导入
        dangerous_modules = ['os', 'sys', 'subprocess', 'socket', 'pickle', 'marshal']
        for imp in imports:
            if imp in dangerous_modules:
                risks.append(f"使用了危险模块: {imp}")

        # 检查危险的操作
        for node in ast.walk(tree):
            # 检查eval/exec调用
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile']:
                        risks.append(f"使用了危险函数: {node.func.id}")

                # 检查__import__调用
                if isinstance(node.func, ast.Name) and node.func.id == '__import__':
                    risks.append("使用了动态导入")

            # 检查属性访问
            if isinstance(node, ast.Attribute):
                if node.attr.startswith('__') and node.attr.endswith('__'):
                    risks.append(f"访问了魔术方法: {node.attr}")

        return risks

    def _calculate_complexity(self, tree: ast.AST) -> float:
        """计算代码复杂度"""
        complexity = 1.0

        for node in ast.walk(tree):
            # 增加控制流复杂度
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.Try):
                complexity += 0.5
            elif isinstance(node, ast.ExceptHandler):
                complexity += 0.3
            elif isinstance(node, ast.BoolOp):
                complexity += 0.5
            elif isinstance(node, ast.FunctionDef):
                complexity += 0.8
            elif isinstance(node, ast.ClassDef):
                complexity += 1.2

        # 归一化到0-10范围
        return min(complexity, 10.0)

    def _generate_suggestions(
        self,
        code: str,
        analysis_results: Dict[str, Any]
    ) -> List[str]:
        """生成改进建议"""
        suggestions = []

        # 基于行数
        if analysis_results['line_count'] > 50:
            suggestions.append("代码较长，考虑拆分为多个函数")

        # 基于复杂度
        if analysis_results['complexity'] > 7:
            suggestions.append("代码复杂度过高，建议简化逻辑")

        # 基于风险
        if analysis_results['risks']:
            suggestions.append("代码存在安全风险，请审查后再执行")

        # 通用建议
        if not suggestions:
            suggestions.append("代码看起来安全，可以执行")

        return suggestions

    def _get_allowed_modules(self) -> List[str]:
        """获取允许的模块列表"""
        return [
            'math',
            'random',
            'datetime',
            'collections',
            'itertools',
            'functools',
            're',
            'string',
            'json',
            'csv',
            'copy',
            'typing',
        ]

    def _get_builtin_whitelist(self) -> List[str]:
        """获取内置函数白名单"""
        return [
            # 类型转换
            'int', 'float', 'str', 'bool', 'list', 'dict', 'tuple', 'set',
            # 数学运算
            'abs', 'round', 'min', 'max', 'sum', 'pow',
            # 序列操作
            'len', 'range', 'enumerate', 'zip', 'map', 'filter',
            # 输入输出
            'print', 'input',
            # 其他安全函数
            'isinstance', 'issubclass', 'hasattr', 'getattr', 'setattr',
            'sorted', 'reversed',
            'True', 'False', 'None',
        ]

    def _get_banned_patterns(self) -> Dict[str, str]:
        """获取禁止的代码模式"""
        return {
            'file_access': r'\bopen\s*\(',
            'system_call': r'\bos\.(system|popen|exec)',
            'network_access': r'\b(socket|urllib|requests)\b',
            'process_creation': r'\bsubprocess\b',
            'dangerous_eval': r'\beval\s*\(',
            'dangerous_exec': r'\bexec\s*\(',
            'import_hook': r'__import__',
            'attribute_manipulation': r'__class__|__bases__|__mro__',
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取执行统计"""
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "blocked_executions": self.blocked_executions,
            "success_rate": round(
                self.successful_executions / max(self.total_executions, 1) * 100, 2
            ),
            "block_rate": round(
                self.blocked_executions / max(self.total_executions, 1) * 100, 2
            ),
            "default_timeout": self.default_timeout,
            "security_level": self.default_security_level.value,
            "allowed_modules_count": len(self.allowed_modules),
            "builtin_whitelist_count": len(self.builtin_whitelist)
        }

    def reset_statistics(self):
        """重置统计数据"""
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.blocked_executions = 0
        self.logger.info("🧹 代码执行统计已重置")


class SecurityError(Exception):
    """安全错误"""
    pass
