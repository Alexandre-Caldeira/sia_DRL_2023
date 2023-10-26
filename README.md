# sia_DRL_2023
# Startup

Guide to open up DDQN navigation map

1. Path
```bash
roscore

cd /home/nero-ia/workspace/QLearningNavigation/QLearningNavigation
roslaunch gazebo_ros QLNavigationCorridor.launch &

cd /home/nero-ia/workspace/QLearningNavigation/QLearningNavigation
source catkin_ws/devel/setup.bash
roslaunch p3dx_gazebo p3dx.launch x:=-4 y:=2
echo 'ready to learn'
```

2. kill roscore
pkill roscore
