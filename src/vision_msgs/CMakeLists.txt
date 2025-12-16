cmake_minimum_required(VERSION 3.8)
project(vision_msgs)

# Find build tools
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Find message dependencies
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rclpy REQUIRED)   # Needed for Python messages

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SegmentationOutput.msg"
  "msg/VisionStatus.msg"
  DEPENDENCIES
    builtin_interfaces
    sensor_msgs
    std_msgs
)

# Export dependencies for Python runtime
ament_export_dependencies(rosidl_default_runtime rclpy)

ament_package()
