#!/bin/bash
# Ubuntu 24.04 部署脚本
# 用于在云环境中设置Dr.Zero Agent

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Dr.Zero Agent Ubuntu 24.04 部署脚本"
echo "=========================================="

# 1. 检查系统要求
echo ""
echo "📋 检查系统要求..."

# 检查Ubuntu版本
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$VERSION_ID" != "24.04" ]]; then
        echo "⚠️  警告：检测到Ubuntu $VERSION_ID，推荐使用Ubuntu 24.04"
    else
        echo "✅ Ubuntu版本: $VERSION_ID"
    fi
fi

# 检查Linux内核版本
KERNEL_VERSION=$(uname -r | cut -d'-' -f1)
KERNEL_MAJOR=$(echo $KERNEL_VERSION | cut -d'.' -f1)
KERNEL_MINOR=$(echo $KERNEL_VERSION | cut -d'.' -f2)

if (( KERNEL_MAJOR > 5 || (KERNEL_MAJOR == 5 && KERNEL_MINOR >= 14) )); then
    echo "✅ Linux内核版本: $KERNEL_VERSION (>= 5.14)"
else
    echo "❌ Linux内核版本: $KERNEL_VERSION (< 5.14)，需要升级"
    exit 1
fi

# 检查CPU核心数
CPU_CORES=$(nproc)
echo "✅ CPU核心数: $CPU_CORES"

# 检查内存
TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_MEM_GB=$((TOTAL_MEM_KB / 1024 / 1024))
echo "✅ 总内存: ${TOTAL_MEM_GB}GB"

if (( TOTAL_MEM_GB < 16 )); then
    echo "⚠️  警告：内存小于16GB，可能影响性能"
fi

# 2. 安装系统依赖
echo ""
echo "📦 安装系统依赖..."
sudo apt-get update
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev

# 3. 创建虚拟环境
echo ""
echo "🐍 创建Python虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 4. 安装Python依赖
echo ""
echo "📦 安装Python依赖..."
pip install --upgrade pip

# 创建requirements.txt（如果不存在）
if [ ! -f requirements.txt ]; then
    cat > requirements.txt << EOF
# Dr.Zero Agent Dependencies
requests>=2.31.0
numpy>=1.24.0
psutil>=5.9.0
scikit-learn>=1.3.0
joblib>=1.3.0
tqdm>=4.66.0
pyyaml>=6.0
python-dateutil>=2.8.0
EOF
fi

pip install -r requirements.txt

# 5. 设置环境变量
echo ""
echo "⚙️  配置环境变量..."

ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" << EOF
# Dr.Zero Agent Environment Configuration

# MiniMax API Configuration
MINIMAX_API_KEY=your_api_key_here

# Environment
ENVIRONMENT=production

# Logging
LOG_LEVEL=INFO

# Model Configuration
MODEL_NAME=MiniMax-M2.7
MODEL_API_URL=https://api.minimax.chat/v1/text/chatcompletion_v2
EOF
    echo "✅ 已创建 .env 文件，请编辑并设置MINIMAX_API_KEY"
else
    echo "✅ .env 文件已存在"
fi

# 6. 创建必要的目录
echo ""
echo "📁 创建必要目录..."
mkdir -p data/long_term_memory
mkdir -p data/experience_replay
mkdir -p data/strategy_optimizer
mkdir -p logs
mkdir -p cache/models

# 7. 设置权限
echo ""
echo "🔒 设置权限..."
chmod +x main.py
chmod 600 .env  # 保护敏感信息

# 8. 测试运行
echo ""
echo "🧪 测试安装..."
python3 -c "import sys; print(f'Python版本: {sys.version}')"
python3 -c "import requests; print('✅ requests模块可用')"
python3 -c "import psutil; print('✅ psutil模块可用')"

# 9. 创建systemd服务文件（可选）
echo ""
read -p "是否创建systemd服务以实现开机自启？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    SERVICE_FILE="/etc/systemd/system/drzero-agent.service"
    sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=Dr.Zero Agent Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python3 $(pwd)/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    echo "✅ systemd服务文件已创建: $SERVICE_FILE"
    echo "   启动服务: sudo systemctl start drzero-agent"
    echo "   启用开机自启: sudo systemctl enable drzero-agent"
    echo "   查看状态: sudo systemctl status drzero-agent"
fi

# 完成
echo ""
echo "=========================================="
echo "✅ 部署完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 编辑 .env 文件，设置MINIMAX_API_KEY"
echo "2. 运行: source venv/bin/activate"
echo "3. 运行: python3 main.py"
echo ""
echo "文档: 查看README.md获取更多信息"
echo ""