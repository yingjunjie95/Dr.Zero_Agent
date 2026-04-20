import os
import time
from core.agent import DrZeroAgent
from config.system_config import SystemConfig
from monitoring.health_checker import HealthChecker


def main():
    """Dr.Zero Agent主入口"""

    print("🚀 启动Dr.Zero Agent...")

    # 1. 系统健康检查
    health_checker = HealthChecker()
    if not health_checker.check_system_health():
        print("❌ 系统健康检查失败，无法启动")
        return

    # 2. 初始化Agent
    print("🔧 初始化Agent组件...")
    agent = DrZeroAgent(config=SystemConfig())

    # 3. 加载长期记忆
    print("🧠 加载记忆系统...")
    agent.memory_system.load_from_persistence()

    # 4. 启动监控
    print("📊 启动监控系统...")
    agent.start_monitoring()

    # 5. 进入主循环
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
            print(f"💡 当前思考: {agent.get_current_state()['thoughts']}\n")

    except KeyboardInterrupt:
        print("\n👋 收到中断信号，正在优雅关闭...")
    finally:
        # 6. 保存状态
        print("💾 保存Agent状态...")
        agent.save_state()
        agent.stop_monitoring()
        print("✅ Dr.Zero Agent已安全关闭")


if __name__ == "__main__":
    main()