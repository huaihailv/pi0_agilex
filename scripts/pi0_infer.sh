#!/bin/bash

SESSION_NAME="openpi_aloha"

# 创建 tmux 会话
tmux new-session -d -s $SESSION_NAME

##### 下方三屏：ROS 系统部分 #####

# 在主窗口中先创建下方区域
tmux split-window -v -t $SESSION_NAME

# 左下角：roscore
tmux select-pane -t $SESSION_NAME:0.1
tmux send-keys 'roscore' C-m

# 中下角：rs_multiple_devices.launch
tmux split-window -h -t $SESSION_NAME:0.1
tmux select-pane -t $SESSION_NAME:0.2
tmux send-keys 'sleep 1 && roslaunch realsense2_camera rs_multiple_devices.launch' C-m

# 右下角：piper_ws 启动
tmux split-window -h -t $SESSION_NAME:0.2
tmux select-pane -t $SESSION_NAME:0.3
tmux send-keys 'sleep 1 && cd ~/cobot_magic/piper_ws/' C-m
tmux send-keys 'source devel/setup.bash' C-m
tmux send-keys 'roslaunch piper start_slave_aloha.launch mode:=1' C-m

##### 上方两个屏幕：conda 模型推理 #####

# 上方左屏：conda activate openpi + pi0_infer.py（延迟2秒）
tmux select-pane -t $SESSION_NAME:0.0
tmux send-keys 'sleep 2 && conda activate openpi' C-m
tmux send-keys 'python /home/agilex/zbl/openpi/openpi-main/scripts/pi0_infer.py' C-m

# 上方右屏：conda activate aloha + agilex_env.py（延迟2秒）
tmux split-window -h -t $SESSION_NAME:0.0
tmux select-pane -t $SESSION_NAME:0.4
tmux send-keys 'sleep 2 && conda activate aloha' C-m
tmux send-keys 'python /home/agilex/zbl/openpi/openpi-main/scripts/agilex_env.py' C-m

# 默认选中最左侧窗口
tmux select-pane -t $SESSION_NAME:0.0

# 附加到 tmux 会话
tmux attach-session -t $SESSION_NAME
