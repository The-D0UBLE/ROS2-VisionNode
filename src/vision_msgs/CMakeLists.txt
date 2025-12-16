cmake_minimum_required(VERSION 3.8)
project(vision_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclpy REQUIRED)  # Python bindings

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SegmentationOutput.msg"
  "msg/VisionStatus.msg"
  DEPENDENCIES builtin_interfaces sensor_msgs std_msgs
)

# Export Python messages to runtime
ament_export_dependencies(rosidl_default_runtime rclpy)

ament_package()
