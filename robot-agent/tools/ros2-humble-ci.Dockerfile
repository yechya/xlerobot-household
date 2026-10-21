ARG ROS_IMAGE=ros:humble-ros-base
FROM ${ROS_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
      nlohmann-json3-dev \
      python3-colcon-common-extensions \
      ros-humble-moveit-ros-planning-interface \
      ros-humble-moveit-servo \
      ros-humble-moveit-task-constructor-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
