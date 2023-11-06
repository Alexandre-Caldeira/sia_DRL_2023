# sia_DRL_2023
# Link to data and results
https://drive.google.com/drive/folders/1Z5hI4lLqsuxULMySTByrWLPpqAdG9Ad8?usp=drive_link
# Startup

- Guide to open up DDQN navigation map

1. On a terminal:
```bash
roscore
```

2. On another terminal window:
```bash
cd /home/nero-ia/workspace/QLearningNavigation/QLearningNavigation;
source catkin_ws/devel/setup.bash;
roslaunch gazebo_ros QLNavigationCorridor.launch &;
roslaunch p3dx_gazebo p3dx.launch x:=-4 y:=2
```
