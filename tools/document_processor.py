"""
文档处理器 - Document Processor
提供多种文档格式的读取、解析和分析功能
支持文本提取、内容分析和结构化输出
专为CPU环境优化，轻量级且高效的文档处理解决方案
"""
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class DocumentType(Enum):
    """文档类型"""
    TXT = "txt"
    PDF = "pdf"
    DOCX = "docx"
    MD = "md"
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    XML = "xml"
    UNKNOWN = "unknown"


@dataclass
class DocumentInfo:
    """文档信息"""
    filename: str
    file_path: str
    document_type: DocumentType
    file_size_bytes: int
    created_time: Optional[str] = None
    modified_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "filename": self.filename,
            "file_path": self.file_path,
            "document_type": self.document_type.value,
            "file_size_bytes": self.file_size_bytes,
            "file_size_kb": round(self.file_size_bytes / 1024, 2),
            "created_time": self.created_time,
            "modified_time": self.modified_time
        }


@dataclass
class DocumentContent:
    """文档内容"""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    sections: List[Dict[str, Any]] = field(default_factory=list)
    word_count: int = 0
    char_count: int = 0
    line_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "text": self.text[:1000],  # 只返回前1000字符
            "text_length": len(self.text),
            "metadata": self.metadata,
            "sections_count": len(self.sections),
            "word_count": self.word_count,
            "char_count": self.char_count,
            "line_count": self.line_count
        }


@dataclass
class AnalysisResult:
    """分析结果"""
    success: bool
    document_info: DocumentInfo
    content: Optional[DocumentContent] = None
    summary: Optional[str] = None
    key_points: List[str] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "document_info": self.document_info.to_dict(),
            "content": self.content.to_dict() if self.content else None,
            "summary": self.summary,
            "key_points": self.key_points[:10],
            "entities_count": len(self.entities),
            "error_message": self.error_message
        }


class DocumentProcessor:
    """
    文档处理器

    功能：
    - 多格式支持：支持TXT、PDF、DOCX、MD、JSON、CSV等格式
    - 文本提取：从各种文档格式中提取纯文本
    - 内容分析：分析文档结构、关键词和实体
    - 摘要生成：生成文档摘要和关键点
    - 元数据提取：提取文档属性和统计信息
    - 安全处理：防止恶意文件和大文件攻击
    """

    def __init__(
        self,
        max_file_size_mb: int = 50,
        supported_types: Optional[List[DocumentType]] = None,
        enable_summary: bool = True,
        enable_entity_extraction: bool = False
    ):
        """
        初始化文档处理器

        Args:
            max_file_size_mb: 最大文件大小（MB）
            supported_types: 支持的文档类型列表
            enable_summary: 是否启用摘要生成
            enable_entity_extraction: 是否启用实体提取
        """
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.supported_types = supported_types or [
            DocumentType.TXT,
            DocumentType.PDF,
            DocumentType.DOCX,
            DocumentType.MD,
            DocumentType.JSON,
            DocumentType.CSV,
            DocumentType.HTML
        ]
        self.enable_summary = enable_summary
        self.enable_entity_extraction = enable_entity_extraction

        self.logger = logging.getLogger("DocumentProcessor")

        # 处理统计
        self.total_processed = 0
        self.successful_processed = 0
        self.failed_processed = 0

        self.logger.info(
            f"✅ 文档处理器初始化完成 - "
            f"最大文件大小: {max_file_size_mb}MB, "
            f"支持格式: {[t.value for t in self.supported_types]}"
        )

    def process_document(
        self,
        file_path: str,
        extract_full_text: bool = True,
        analyze_content: bool = True
    ) -> Dict[str, Any]:
        """
        处理文档

        Args:
            file_path: 文档文件路径
            extract_full_text: 是否提取完整文本
            analyze_content: 是否分析内容

        Returns:
            处理结果字典
        """
        self.total_processed += 1

        try:
            # 验证文件
            self._validate_file(file_path)

            # 获取文档信息
            doc_info = self._get_document_info(file_path)

            # 检测文档类型
            doc_type = self._detect_document_type(file_path)

            if doc_type not in self.supported_types:
                raise ValueError(f"不支持的文档类型: {doc_type.value}")

            # 提取文本内容
            content = None
            if extract_full_text:
                content = self._extract_text(file_path, doc_type)

            # 分析内容
            summary = None
            key_points = []
            entities = []

            if analyze_content and content:
                summary = self._generate_summary(content.text)
                key_points = self._extract_key_points(content.text)

                if self.enable_entity_extraction:
                    entities = self._extract_entities(content.text)

            result = AnalysisResult(
                success=True,
                document_info=doc_info,
                content=content,
                summary=summary,
                key_points=key_points,
                entities=entities
            )

            self.successful_processed += 1
            self.logger.info(
                f"✅ 文档处理成功: {doc_info.filename}, "
                f"类型: {doc_type.value}, "
                f"字数: {content.word_count if content else 0}"
            )

            return result.to_dict()

        except Exception as e:
            self.failed_processed += 1
            self.logger.error(f"❌ 文档处理失败: {str(e)}")

            # 尝试获取文档信息
            try:
                doc_info = self._get_document_info(file_path)
            except:
                doc_info = DocumentInfo(
                    filename=os.path.basename(file_path),
                    file_path=file_path,
                    document_type=DocumentType.UNKNOWN,
                    file_size_bytes=0
                )

            error_result = AnalysisResult(
                success=False,
                document_info=doc_info,
                error_message=str(e)
            )

            return error_result.to_dict()

    def process_text(
        self,
        text: str,
        title: str = "Untitled",
        analyze: bool = True
    ) -> Dict[str, Any]:
        """
        处理纯文本

        Args:
            text: 文本内容
            title: 文本标题
            analyze: 是否分析内容

        Returns:
            处理结果字典
        """
        self.total_processed += 1

        try:
            # 创建文档信息
            doc_info = DocumentInfo(
                filename=f"{title}.txt",
                file_path="",
                document_type=DocumentType.TXT,
                file_size_bytes=len(text.encode('utf-8'))
            )

            # 创建内容对象
            content = self._create_content_object(text)

            # 分析内容
            summary = None
            key_points = []
            entities = []

            if analyze:
                summary = self._generate_summary(text)
                key_points = self._extract_key_points(text)

                if self.enable_entity_extraction:
                    entities = self._extract_entities(text)

            result = AnalysisResult(
                success=True,
                document_info=doc_info,
                content=content,
                summary=summary,
                key_points=key_points,
                entities=entities
            )

            self.successful_processed += 1
            return result.to_dict()

        except Exception as e:
            self.failed_processed += 1
            self.logger.error(f"❌ 文本处理失败: {str(e)}")

            error_result = AnalysisResult(
                success=False,
                document_info=DocumentInfo(
                    filename=f"{title}.txt",
                    file_path="",
                    document_type=DocumentType.TXT,
                    file_size_bytes=0
                ),
                error_message=str(e)
            )

            return error_result.to_dict()

    def _validate_file(self, file_path: str):
        """验证文件"""
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 检查是否为文件
        if not os.path.isfile(file_path):
            raise ValueError(f"不是有效文件: {file_path}")

        # 检查文件大小
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size_bytes:
            raise ValueError(
                f"文件过大: {file_size / (1024*1024):.2f}MB "
                f"(最大: {self.max_file_size_bytes / (1024*1024)}MB)"
            )

        # 检查文件可读性
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"文件不可读: {file_path}")

    def _get_document_info(self, file_path: str) -> DocumentInfo:
        """获取文档信息"""
        stat = os.stat(file_path)

        return DocumentInfo(
            filename=os.path.basename(file_path),
            file_path=file_path,
            document_type=DocumentType.UNKNOWN,  # 稍后更新
            file_size_bytes=stat.st_size,
            created_time=datetime.fromtimestamp(stat.st_ctime).isoformat(),
            modified_time=datetime.fromtimestamp(stat.st_mtime).isoformat()
        )

    def _detect_document_type(self, file_path: str) -> DocumentType:
        """检测文档类型"""
        extension = Path(file_path).suffix.lower().lstrip('.')

        type_map = {
            'txt': DocumentType.TXT,
            'pdf': DocumentType.PDF,
            'docx': DocumentType.DOCX,
            'md': DocumentType.MD,
            'markdown': DocumentType.MD,
            'json': DocumentType.JSON,
            'csv': DocumentType.CSV,
            'html': DocumentType.HTML,
            'htm': DocumentType.HTML,
            'xml': DocumentType.XML
        }

        return type_map.get(extension, DocumentType.UNKNOWN)

    def _extract_text(self, file_path: str, doc_type: DocumentType) -> DocumentContent:
        """提取文档文本"""
        try:
            if doc_type == DocumentType.TXT:
                text = self._read_txt_file(file_path)
            elif doc_type == DocumentType.MD:
                text = self._read_txt_file(file_path)
            elif doc_type == DocumentType.PDF:
                text = self._read_pdf_file(file_path)
            elif doc_type == DocumentType.DOCX:
                text = self._read_docx_file(file_path)
            elif doc_type == DocumentType.JSON:
                text = self._read_json_file(file_path)
            elif doc_type == DocumentType.CSV:
                text = self._read_csv_file(file_path)
            elif doc_type == DocumentType.HTML:
                text = self._read_html_file(file_path)
            else:
                raise ValueError(f"不支持的文档类型: {doc_type.value}")

            return self._create_content_object(text)

        except Exception as e:
            self.logger.error(f"文本提取失败: {str(e)}")
            raise

    def _read_txt_file(self, file_path: str) -> str:
        """读取TXT文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _read_pdf_file(self, file_path: str) -> str:
        """读取PDF文件"""
        try:
            import PyPDF2

            text_parts = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)

            return '\n\n'.join(text_parts)

        except ImportError:
            self.logger.warning("⚠️ PyPDF2未安装，无法处理PDF文件")
            raise ImportError("需要安装PyPDF2: pip install PyPDF2")

        except Exception as e:
            self.logger.error(f"PDF读取失败: {str(e)}")
            raise

    def _read_docx_file(self, file_path: str) -> str:
        """读取DOCX文件"""
        try:
            from docx import Document

            doc = Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]

            return '\n\n'.join(paragraphs)

        except ImportError:
            self.logger.warning("⚠️ python-docx未安装，无法处理DOCX文件")
            raise ImportError("需要安装python-docx: pip install python-docx")

        except Exception as e:
            self.logger.error(f"DOCX读取失败: {str(e)}")
            raise

    def _read_json_file(self, file_path: str) -> str:
        """读取JSON文件"""
        import json

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 格式化JSON为可读文本
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _read_csv_file(self, file_path: str) -> str:
        """读取CSV文件"""
        import csv

        rows = []
        with open(file_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)

            for row in csv_reader:
                rows.append(', '.join(row))

        return '\n'.join(rows)

    def _read_html_file(self, file_path: str) -> str:
        """读取HTML文件"""
        try:
            from bs4 import BeautifulSoup

            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')

            # 提取文本
            text = soup.get_text(separator='\n', strip=True)

            return text

        except ImportError:
            self.logger.warning("⚠️ beautifulsoup4未安装，使用简单HTML解析")
            return self._simple_html_parse(file_path)

        except Exception as e:
            self.logger.error(f"HTML读取失败: {str(e)}")
            raise

    def _simple_html_parse(self, file_path: str) -> str:
        """简单HTML解析（备用方案）"""
        import re

        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', html_content)

        # 清理空白
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        return '\n'.join(lines)

    def _create_content_object(self, text: str) -> DocumentContent:
        """创建内容对象"""
        # 计算统计信息
        word_count = len(text.split())
        char_count = len(text)
        line_count = len(text.splitlines())

        # 提取元数据
        metadata = {
            "encoding": "utf-8",
            "language": self._detect_language(text)
        }

        # 提取章节（如果有明显的标题）
        sections = self._extract_sections(text)

        return DocumentContent(
            text=text,
            metadata=metadata,
            sections=sections,
            word_count=word_count,
            char_count=char_count,
            line_count=line_count
        )

    def _detect_language(self, text: str) -> str:
        """检测语言"""
        # 简单的语言检测
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text.replace(' ', ''))

        if total_chars == 0:
            return "unknown"

        chinese_ratio = chinese_chars / total_chars

        if chinese_ratio > 0.3:
            return "zh-CN"
        else:
            return "en"

    def _extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """提取章节"""
        sections = []
        lines = text.split('\n')

        current_section = {"title": "Introduction", "content": [], "level": 0}

        for line in lines:
            stripped = line.strip()

            # 检测Markdown标题
            if stripped.startswith('#'):
                # 保存之前的章节
                if current_section["content"]:
                    sections.append({
                        "title": current_section["title"],
                        "content": '\n'.join(current_section["content"]),
                        "level": current_section["level"]
                    })

                # 开始新章节
                level = len(stripped) - len(stripped.lstrip('#'))
                title = stripped.lstrip('#').strip()

                current_section = {
                    "title": title,
                    "content": [],
                    "level": level
                }
            else:
                if stripped:
                    current_section["content"].append(stripped)

        # 添加最后一个章节
        if current_section["content"]:
            sections.append({
                "title": current_section["title"],
                "content": '\n'.join(current_section["content"]),
                "level": current_section["level"]
            })

        return sections[:20]  # 最多返回20个章节

    def _generate_summary(self, text: str) -> Optional[str]:
        """生成摘要"""
        if not self.enable_summary or not text:
            return None

        try:
            # 简单摘要：提取前几个句子
            sentences = text.replace('\n', ' ').split('。')

            if len(sentences) <= 3:
                return text[:500]

            # 取前3个句子
            summary_sentences = sentences[:3]
            summary = '。'.join(summary_sentences) + '。'

            # 限制长度
            if len(summary) > 500:
                summary = summary[:500] + '...'

            return summary

        except Exception as e:
            self.logger.error(f"摘要生成失败: {str(e)}")
            return None

    def _extract_key_points(self, text: str) -> List[str]:
        """提取关键点"""
        if not text:
            return []

        try:
            # 简单关键点提取：提取包含关键词的句子
            keywords = ["重要", "关键", "主要", "首先", "其次", "最后",
                       "总结", "结论", "因此", "所以", "然而"]

            sentences = text.replace('\n', ' ').split('。')
            key_points = []

            for sentence in sentences:
                sentence = sentence.strip()

                if any(keyword in sentence for keyword in keywords):
                    if sentence and len(sentence) > 10:
                        key_points.append(sentence + '。')

                        if len(key_points) >= 5:
                            break

            # 如果没有找到关键点，返回前几个句子
            if not key_points and sentences:
                key_points = [s.strip() + '。' for s in sentences[:3] if s.strip()]

            return key_points

        except Exception as e:
            self.logger.error(f"关键点提取失败: {str(e)}")
            return []

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """提取实体"""
        if not self.enable_entity_extraction or not text:
            return []

        try:
            # 这里可以集成NER模型，暂时返回空列表
            self.logger.debug("实体提取功能待实现")
            return []

        except Exception as e:
            self.logger.error(f"实体提取失败: {str(e)}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计"""
        return {
            "total_processed": self.total_processed,
            "successful_processed": self.successful_processed,
            "failed_processed": self.failed_processed,
            "success_rate": round(
                self.successful_processed / max(self.total_processed, 1) * 100, 2
            ),
            "supported_formats": [t.value for t in self.supported_types],
            "max_file_size_mb": self.max_file_size_bytes / (1024 * 1024)
        }
