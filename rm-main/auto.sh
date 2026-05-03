#!/bin/bash

# ==========================================
# 监控配置项 (使用前请修改)
# ==========================================

# 1. 程序的唯一标识，用于检查它是否存活
APP_NAME="autotest"

# 2. 程序的启动命令 (当检测到程序死亡时，执行此命令)
# 建议写绝对路径，避免环境变量问题，例如: "/usr/bin/python3 /home/user/my_app.py"
START_CMD="/home/king/test/autotest"

# 3. 轮询间隔 (单位：秒，例如 10秒 检查一次)
CHECK_INTERVAL=5

# 4. 监控程序的日志文件路径 (记录什么时候挂掉、什么时候重启)
MONITOR_LOG="/home/king/test/log/monitor_guard.log"

# 5. 业务程序的输出日志文件
APP_LOG="/home/king/test/log/app.log"

# ==========================================
# 核心监控逻辑
# ==========================================

echo "启动监控器，开始守护 $APP_NAME ..."
echo "轮询间隔: $CHECK_INTERVAL 秒"
echo "监控日志: $MONITOR_LOG"

# 开启死循环进行无限轮询
while true; do
    # 查找进程并提取 PID
    PID=$(ps -ef | grep "$APP_NAME" | grep -v grep | awk '{print $2}')
    
    # 检查 PID 是否为空
    if [ -z "$PID" ]; then
        # 获取当前时间
        NOW=$(date +"%Y-%m-%d %H:%M:%S")
        
        echo "[$NOW] ⚠️ 警告: 检测到 $APP_NAME 未运行，准备拉起..." | tee -a "$MONITOR_LOG"
        
        # 执行启动命令，放入后台，并将输出重定向
        nohup $START_CMD > "$APP_LOG" 2>&1 &
        
        # 暂停2秒，给程序一点启动时间
        sleep 2
        
        # 再次检查是否拉起成功
        NEW_PID=$(ps -ef | grep "$APP_NAME" | grep -v grep | awk '{print $2}')
        if [ ! -z "$NEW_PID" ]; then
            echo "[$NOW] ✅ 恢复: $APP_NAME 拉起成功，新 PID: $NEW_PID" | tee -a "$MONITOR_LOG"
        else
            echo "[$NOW] ❌ 错误: $APP_NAME 拉起失败，请检查程序本身是否有报错！" | tee -a "$MONITOR_LOG"
        fi
    fi
    
    # 等待设定的间隔时间后，进行下一次检测
    sleep $CHECK_INTERVAL
done