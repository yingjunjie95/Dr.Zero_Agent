"""
搜索任务评测脚本 - Search Evaluation Script
运行10个标准化测试用例，生成综合评测报告
"""
import time
import json
from datetime import datetime
from typing import List, Dict, Any
from benchmarks.test_queries import STANDARD_TEST_QUERIES, SearchEvaluationMetrics, TestScenario
from tools.web_search import WebSearchTool, SearchEngine


class SearchBenchmarkEvaluator:
    """搜索基准评测器"""

    def __init__(self):
        self.search_tool = WebSearchTool(
            default_engine=SearchEngine.DUCKDUCKGO,
            max_results=5,
            enable_multi_source=True,
            enable_reranking=True,
            enable_cache=False
        )
        self.results: List[Dict[str, Any]] = []

    def run_all_tests(self, output_file: str = None) -> Dict[str, Any]:
        """运行所有测试"""
        print("=" * 60)
        print("🚀 Dr.Zero Agent - 信息搜索技能评测")
        print("=" * 60)
        print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📋 测试用例数: {len(STANDARD_TEST_QUERIES)}")
        print()

        overall_start = time.time()

        for test_query in STANDARD_TEST_QUERIES:
            result = self._run_single_test(test_query)
            self.results.append(result)

            print(f"✅ 测试 {test_query.id}/10 完成 - 得分: {result['score']['total_score']:.2f}")
            print()

        overall_time = time.time() - overall_start

        report = self._generate_report(overall_time)

        if output_file:
            self._save_report(report, output_file)

        return report

    def _run_single_test(self, test_query) -> Dict[str, Any]:
        """运行单个测试"""
        scenario_emoji = {
            TestScenario.POLICY_REGULATION: "📜",
            TestScenario.INDUSTRY_INFO: "🏭",
            TestScenario.ACADEMIC_LITERATURE: "📚"
        }

        emoji = scenario_emoji.get(test_query.scenario, "🔍")

        print(f"{emoji} 测试 #{test_query.id}: {test_query.query}")
        print(f"   场景: {test_query.scenario.value} | 难度: {test_query.difficulty}")

        test_start = time.time()

        try:
            search_result = self.search_tool.search(
                query=test_query.query,
                num_results=5,
                structured_output=True
            )

            test_time = time.time() - test_start

            results_list = search_result.get('results', [])

            completeness = SearchEvaluationMetrics.evaluate_completeness(
                results_list, test_query.keywords
            )
            diversity = SearchEvaluationMetrics.evaluate_source_diversity(results_list)
            relevance = SearchEvaluationMetrics.evaluate_relevance(results_list, test_query.query)
            timeliness = SearchEvaluationMetrics.evaluate_timeliness(results_list)

            score = SearchEvaluationMetrics.calculate_comprehensive_score(
                completeness=completeness,
                diversity=diversity,
                relevance=relevance,
                timeliness=timeliness,
                result_count=len(results_list),
                response_time=test_time
            )

            print(f"   📊 综合得分: {score['total_score']:.2f}/100")
            print(f"   🎯 完整性: {score['completeness']:.1f}%")
            print(f"   🔗 来源多样性: {score['source_diversity']:.1f}%")
            print(f"   ⭐ 相关性: {score['relevance']:.1f}%")
            print(f"   ⏱️ 时效性: {score['timeliness']:.1f}%")
            print(f"   ⚡ 效率: {score['efficiency']:.1f}%")
            print(f"   🔍 结果数: {score['result_count']}")
            print(f"   ⏰ 响应时间: {score['response_time_ms']:.0f}ms")

            return {
                'test_id': test_query.id,
                'scenario': test_query.scenario.value,
                'query': test_query.query,
                'difficulty': test_query.difficulty,
                'score': score,
                'structured_output': search_result,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            test_time = time.time() - test_start
            print(f"   ❌ 测试失败: {str(e)}")

            return {
                'test_id': test_query.id,
                'scenario': test_query.scenario.value,
                'query': test_query.query,
                'difficulty': test_query.difficulty,
                'score': {
                    'total_score': 0.0,
                    'completeness': 0.0,
                    'source_diversity': 0.0,
                    'relevance': 0.0,
                    'timeliness': 0.0,
                    'efficiency': 0.0,
                    'result_count': 0,
                    'response_time_ms': test_time * 1000
                },
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _generate_report(self, total_time: float) -> Dict[str, Any]:
        """生成评测报告"""
        successful_tests = [r for r in self.results if r['success']]
        failed_tests = [r for r in self.results if not r['success']]

        avg_score = (
            sum(r['score']['total_score'] for r in successful_tests) / len(successful_tests)
            if successful_tests else 0
        )

        scenario_scores = {}
        for result in self.results:
            scenario = result['scenario']
            if scenario not in scenario_scores:
                scenario_scores[scenario] = []
            scenario_scores[scenario].append(result['score']['total_score'])

        scenario_avg = {
            scenario: sum(scores) / len(scores)
            for scenario, scores in scenario_scores.items()
        }

        report = {
            'summary': {
                'total_tests': len(self.results),
                'successful': len(successful_tests),
                'failed': len(failed_tests),
                'success_rate': round(len(successful_tests) / len(self.results) * 100, 2),
                'average_score': round(avg_score, 2),
                'total_time_seconds': round(total_time, 2)
            },
            'scenario_performance': scenario_avg,
            'test_results': self.results,
            'timestamp': datetime.now().isoformat()
        }

        print("\n" + "=" * 60)
        print("📊 评测报告总结")
        print("=" * 60)
        print(f"✅ 成功: {len(successful_tests)}/{len(self.results)}")
        print(f"❌ 失败: {len(failed_tests)}/{len(self.results)}")
        print(f"📈 成功率: {report['summary']['success_rate']:.1f}%")
        print(f"🎯 平均得分: {avg_score:.2f}/100")
        print(f"⏱️ 总耗时: {total_time:.2f}秒")

        print("\n📂 各场景表现:")
        for scenario, score in scenario_avg.items():
            print(f"   {scenario}: {score:.2f}/100")

        print("\n📝 详细结果已保存到 evaluations 目录")
        print("=" * 60)

        return report

    def _save_report(self, report: Dict[str, Any], output_file: str):
        """保存评测报告"""
        import os

        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n💾 报告已保存: {output_file}")


def main():
    """主函数"""
    evaluator = SearchBenchmarkEvaluator()

    output_file = "evaluations/search_benchmark_report.json"

    report = evaluator.run_all_tests(output_file=output_file)

    return report


if __name__ == "__main__":
    main()