# Dr.Zero Agent - 自主认知智能体

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)
![Platform](https://img.shields.io/badge/platform-Ubuntu%2024.04-lightgrey.svg)

**具有元认知能力的自主AI智能体 | 自适应学习 | 三层记忆架构**

</div>

---

## 📖 项目简介

Dr.Zero Agent 是一个具有自主认知能力的智能体系统，采用类人认知架构设计，具备感知、认知、记忆、行动和学习的完整闭环。系统基于 **MiniMax-M2.7** 大语言模型，运行于 **Ubuntu 24.04** 云环境（4 vCPU, 16GB RAM），实现了真正的自主性、反应性、主动性和适应性。

### ✨ 核心特性

- 🧠 **三层记忆架构** - 感官记忆、工作记忆、长期记忆的类人记忆系统
- 🔍 **元认知能力** - 自我评估、风险控制、置信度校准
- 🎯 **自主决策** - 动态策略选择、多方案备选、风险感知
- 📚 **持续学习** - 经验回放、策略优化、失败分析
- 🛡️ **安全沙箱** - 工具执行隔离、权限控制、输入验证
- 🌐 **多工具集成** - 网络搜索、文档处理、API调用、代码执行
- ⚡ **性能优化** - CPU优化、缓存机制、资源监控

---

## 🏗️ 系统架构

![](C:\Users\junji\OneDrive\Documents\项目\加文件\mermaid-diagram.png)

---

## 🚀 快速开始

### 前置要求

- **操作系统**: Ubuntu 24.04 LTS
- **Linux内核**: >= 5.14
- **Python**: >= 3.10
- **硬件**: 4 vCPU, 16GB RAM (推荐)
- **API密钥**: [MiniMax平台](https://platform.minimaxi.com) 注册的API密钥

### 安装步骤

#### 1. 克隆项目

```bash
git clone <repository-url> && cd Dr.Zero_Agent
```

#### 2. 运行自动部署脚本

```bash 
chmod +x deploy/setup_ubuntu.sh ./deploy/setup_ubuntu.sh
```

部署脚本将自动：
- ✅ 检查系统要求（Ubuntu版本、内核、CPU、内存）
- ✅ 安装系统依赖和Python包
- ✅ 创建虚拟环境
- ✅ 生成配置文件模板
- ✅ 设置必要目录和权限

#### 3. 配置API密钥

编辑 `.env` 文件：

```bash 
nano .env
```

```env
MINIMAX_API_KEY=your_actual_api_key_here
```

> ⚠️ **重要**: 从 [MiniMax平台](https://platform.minimaxi.com) 获取API密钥

#### 4. 启动Agent


```bash 
source venv/bin/activate python3 main.py
```

---

## 💬 使用示例

### 交互式对话
