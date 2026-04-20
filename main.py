import os
import time
from core.agent import DrZeroAgent
from config.system_config import SystemConfig
from monitoring.health_checker import HealthChecker


def main():
    """Dr.Zero Agent主入口"""

    print("🚀 启动Dr.Zero Agent...")

    # 1. 初始化Agent
    print("🔧 初始化Agent组件...")
    agent = DrZeroAgent(config=SystemConfig())

    # 2. 加载长期记忆
    print("🧠 加载记忆系统...")
    agent.memory_system.load_from_persistence()

    # 3. 初始化并启动健康检查器
    print("📊 启动监控系统...")
    health_checker = HealthChecker()

    # 注册Agent核心组件的健康检查
    health_checker.register_component(
        name="agent_core",
        health_check_func=lambda: {
            "status": "healthy" if agent.is_active else "unavailable",
            "response_time_ms": 0,
            "details": "Agent核心运行正常"
        },
        critical=True
    )

    health_checker.start_monitoring()
    agent.start_monitoring()

    # 4. 进入主循环
    print("✅ Dr.Zero Agent启动成功！")
    print(f"📋 当前能力: {', '.join(agent.get_capabilities())}")
    print(f"🎯 随时准备为您服务\n")

    try:
        while True:
            # 获取用户输入
            user_input = input("👤 用户: ")

            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("👋 Dr.Zero Agent正在关闭...")
                break

            # Agent处理
            start_time = time.time()
            response = agent.process_input(user_input)
            processing_time = time.time() - start_time

            # 显示结果
            print(f"🤖 Dr.Zero: {response}")
            print(f"⏱️ 处理时间: {processing_time:.2f}秒")

            # 优化：显示简化的思考信息
            current_state = agent.get_current_state()
            thoughts = current_state.get('thoughts')
            if thoughts:
                cognition = thoughts.get('cognition', {})
                metacognitive = cognition.get('metacognitive_assessment', {})

                print(f"💡 置信度: {metacognitive.get('overall_confidence', 0):.2%}")
                print(f"🎯 认知状态: {metacognitive.get('cognitive_state', 'unknown')}")
                print(f"⚠️ 风险等级: {metacognitive.get('risk_level', 'unknown')}")

                # 只在需要时显示详细信息
                if metacognitive.get('requires_clarification'):
                    print(f"❓ 需要澄清: {metacognitive.get('clarification_points', [])}")

            print()

    except KeyboardInterrupt:
        print("\n👋 收到中断信号，正在优雅关闭...")
    finally:
        # 5. 保存状态
        print("💾 保存Agent状态...")
        agent.save_state()
        agent.stop_monitoring()
        health_checker.stop_monitoring()
        print("✅ Dr.Zero Agent已安全关闭")


if __name__ == "__main__":
    main()