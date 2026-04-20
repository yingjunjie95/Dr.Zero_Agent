"""
信息搜索技能 Demo 演示
展示Dr.Zero Agent的搜索能力和结构化输出
"""
import time
from tools.web_search import WebSearchTool, SearchEngine
from benchmarks.test_queries import STANDARD_TEST_QUERIES


def demo_structured_search():
    """演示结构化搜索"""
    print("=" * 70)
    print("🎯 Dr.Zero Agent - 信息搜索技能 Demo")
    print("=" * 70)
    print()

    search_tool = WebSearchTool(
        default_engine=SearchEngine.DUCKDUCKGO,
        max_results=5,
        enable_multi_source=True,
        enable_reranking=True,
        enable_cache=False
    )

    demo_queries = [
        ("政策法规", STANDARD_TEST_QUERIES[0].query),
        ("行业信息", STANDARD_TEST_QUERIES[3].query),
        ("学术文献", STANDARD_TEST_QUERIES[6].query),
    ]

    for scenario, query in demo_queries:
        print(f"📌 场景: {scenario}")
        print(f"🔍 查询: {query}")
        print("-" * 70)

        start_time = time.time()

        result = search_tool.search(
            query=query,
            num_results=5,
            structured_output=True
        )

        elapsed = time.time() - start_time

        if result.get('success'):
            print(f"✅ 搜索成功 - 找到 {result.get('result_count', 0)} 条结果")
            print(f"⏱️ 耗时: {elapsed:.2f}秒")
            print(f"🔧 引擎: {result.get('search_metadata', {}).get('engine', 'unknown')}")
            print()

            print("📄 搜索结果:")
            for item in result.get('results', []):
                print(f"  {item['rank']}. {item['title']}")
                print(f"     来源: {item['source']}")
                print(f"     相关度: {item['relevance_score']:.2%}")
                if item.get('published_date'):
                    print(f"     发布时间: {item['published_date']}")
                print(f"     {item['snippet'][:100]}...")
                print()
        else:
            print(f"❌ 搜索失败: {result.get('error', '未知错误')}")

        print("=" * 70)
        print()


def demo_token_efficiency():
    """演示Token效率优化"""
    print("=" * 70)
    print("💰 Token 效率优化 Demo")
    print("=" * 70)
    print()

    search_tool = WebSearchTool(
        default_engine=SearchEngine.DUCKDUCKGO,
        max_results=3,
        enable_multi_source=True,
        enable_reranking=True,
        enable_cache=True
    )

    query = "2024年人工智能发展趋势"

    print(f"🔍 查询: {query}")
    print()

    print("1️⃣ 第一次搜索（缓存未命中）")
    start = time.time()
    result1 = search_tool.search(query=query, num_results=3)
    time1 = time.time() - start
    print(f"   耗时: {time1:.2f}秒")
    print(f"   结果数: {result1.get('result_count', 0)}")
    print()

    print("2️⃣ 第二次搜索（缓存命中）")
    start = time.time()
    result2 = search_tool.search(query=query, num_results=3)
    time2 = time.time() - start
    print(f"   耗时: {time2:.2f}秒")
    print(f"   结果数: {result2.get('result_count', 0)}")
    print(f"   💾 节省: {((time1 - time2) / time1 * 100):.1f}% 时间")
    print()

    stats = search_tool.get_statistics()
    print("📊 搜索统计:")
    print(f"   总搜索次数: {stats['total_searches']}")
    print(f"   缓存命中: {stats['cache_hits']}")
    print(f"   成功率: {stats['success_rate']}%")
    print()
    print("=" * 70)


def main():
    """运行所有Demo"""
    print("\n" + "🚀" * 35 + "\n")
    print("欢迎体验 Dr.Zero Agent 信息搜索技能")
    print()

    demo_structured_search()
    demo_token_efficiency()

    print("\n✨ Demo 演示完成！")
    print("\n💡 提示: 运行 evaluations/run_evaluation.py 进行完整评测")


if __name__ == "__main__":
    main()