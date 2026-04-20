"""
标准化测试基准 - Standardized Test Benchmarks
针对信息搜索任务的10个固定测试用例
覆盖：政策法规、行业信息、学术文献三大场景
"""
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class TestScenario(Enum):
    """测试场景类型"""
    POLICY_REGULATION = "政策法规"
    INDUSTRY_INFO = "行业信息"
    ACADEMIC_LITERATURE = "学术文献"


@dataclass
class TestQuery:
    """测试查询"""
    id: int
    scenario: TestScenario
    query: str
    description: str
    expected_source_types: List[str]
    difficulty: str
    keywords: List[str]


# 10个标准化测试查询
STANDARD_TEST_QUERIES = [
    TestQuery(
        id=1,
        scenario=TestScenario.POLICY_REGULATION,
        query="2024年最新个人所得税优惠政策",
        description="搜索国家最新个人所得税减免政策",
        expected_source_types=["政府网站", "权威媒体"],
        difficulty="medium",
        keywords=["个人所得税", "优惠政策", "2024"]
    ),
    TestQuery(
        id=2,
        scenario=TestScenario.POLICY_REGULATION,
        query="数据安全法实施细则解读",
        description="查找数据安全法相关实施规定",
        expected_source_types=["政府网站", "法律数据库"],
        difficulty="hard",
        keywords=["数据安全法", "实施细则", "解读"]
    ),
    TestQuery(
        id=3,
        scenario=TestScenario.POLICY_REGULATION,
        query="新能源汽车补贴政策2024",
        description="了解最新新能源汽车购置补贴标准",
        expected_source_types=["政府网站", "行业新闻"],
        difficulty="medium",
        keywords=["新能源汽车", "补贴", "2024"]
    ),
    TestQuery(
        id=4,
        scenario=TestScenario.INDUSTRY_INFO,
        query="人工智能行业最新发展趋势",
        description="搜索AI行业最新技术趋势和市场动态",
        expected_source_types=["行业报告", "新闻媒体", "研究机构"],
        difficulty="medium",
        keywords=["人工智能", "发展趋势", "AI"]
    ),
    TestQuery(
        id=5,
        scenario=TestScenario.INDUSTRY_INFO,
        query="芯片半导体产业投资前景分析",
        description="查找芯片半导体行业投资机会分析",
        expected_source_types=["行业报告", "投资分析"],
        difficulty="hard",
        keywords=["芯片", "半导体", "投资前景"]
    ),
    TestQuery(
        id=6,
        scenario=TestScenario.INDUSTRY_INFO,
        query="跨境电商平台运营策略",
        description="了解跨境电商最新运营模式和策略",
        expected_source_types=["行业博客", "电商论坛", "案例分享"],
        difficulty="easy",
        keywords=["跨境电商", "运营策略", "平台"]
    ),
    TestQuery(
        id=7,
        scenario=TestScenario.ACADEMIC_LITERATURE,
        query="大语言模型幻觉问题研究综述",
        description="搜索关于LLM幻觉问题的学术论文",
        expected_source_types=["学术数据库", "论文网站"],
        difficulty="hard",
        keywords=["大语言模型", "幻觉", "研究综述"]
    ),
    TestQuery(
        id=8,
        scenario=TestScenario.ACADEMIC_LITERATURE,
        query="强化学习在机器人控制中的应用",
        description="查找强化学习用于机器人控制的学术文献",
        expected_source_types=["学术数据库", "会议论文"],
        difficulty="medium",
        keywords=["强化学习", "机器人", "应用"]
    ),
    TestQuery(
        id=9,
        scenario=TestScenario.ACADEMIC_LITERATURE,
        query="区块链技术在供应链金融中的应用研究",
        description="搜索区块链在供应链金融领域的学术成果",
        expected_source_types=["学术数据库", "期刊论文"],
        difficulty="medium",
        keywords=["区块链", "供应链金融", "应用"]
    ),
    TestQuery(
        id=10,
        scenario=TestScenario.POLICY_REGULATION,
        query="碳排放交易政策最新动态",
        description="了解全国碳排放权交易市场政策更新",
        expected_source_types=["政府网站", "环保部门", "权威媒体"],
        difficulty="hard",
        keywords=["碳排放", "交易政策", "最新动态"]
    ),
]


class SearchEvaluationMetrics:
    """搜索评估指标"""

    @staticmethod
    def evaluate_completeness(results: List[Dict[str, Any]], expected_keywords: List[str]) -> float:
        """
        评估结果完整性（0-1）
        计算覆盖了多少期望关键词
        """
        if not expected_keywords:
            return 0.0

        matched_keywords = 0
        for keyword in expected_keywords:
            for result in results:
                title = result.get('title', '').lower()
                snippet = result.get('snippet', '').lower()
                if keyword.lower() in title or keyword.lower() in snippet:
                    matched_keywords += 1
                    break

        return matched_keywords / len(expected_keywords)

    @staticmethod
    def evaluate_source_diversity(results: List[Dict[str, Any]]) -> float:
        """
        评估来源多样性（0-1）
        计算不同来源的占比
        """
        if not results:
            return 0.0

        unique_sources = set()
        for result in results:
            source = result.get('source', 'Unknown')
            unique_sources.add(source)

        return min(len(unique_sources) / 3, 1.0)

    @staticmethod
    def evaluate_relevance(results: List[Dict[str, Any]], query: str) -> float:
        """
        评估相关性（0-1）
        基于平均相关度评分
        """
        if not results:
            return 0.0

        avg_relevance = sum(result.get('relevance_score', 0) for result in results) / len(results)
        return avg_relevance

    @staticmethod
    def evaluate_timeliness(results: List[Dict[str, Any]]) -> float:
        """
        评估时效性（0-1）
        检查是否有发布时间信息
        """
        if not results:
            return 0.0

        results_with_date = sum(
            1 for result in results
            if result.get('published_date') is not None
        )

        return results_with_date / len(results)

    @staticmethod
    def calculate_comprehensive_score(
        completeness: float,
        diversity: float,
        relevance: float,
        timeliness: float,
        result_count: int,
        response_time: float
    ) -> Dict[str, Any]:
        """
        计算综合评分

        Returns:
            包含各项得分和总分的字典
        """
        weights = {
            'completeness': 0.30,
            'diversity': 0.20,
            'relevance': 0.25,
            'timeliness': 0.15,
            'efficiency': 0.10
        }

        efficiency = max(0, 1.0 - (response_time / 10.0))

        total_score = (
            completeness * weights['completeness'] +
            diversity * weights['diversity'] +
            relevance * weights['relevance'] +
            timeliness * weights['timeliness'] +
            efficiency * weights['efficiency']
        )

        return {
            'completeness': round(completeness * 100, 2),
            'source_diversity': round(diversity * 100, 2),
            'relevance': round(relevance * 100, 2),
            'timeliness': round(timeliness * 100, 2),
            'efficiency': round(efficiency * 100, 2),
            'result_count': result_count,
            'response_time_ms': round(response_time * 1000, 2),
            'total_score': round(total_score * 100, 2)
        }