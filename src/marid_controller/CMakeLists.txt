cmake_minimum_required(VERSION 3.8)
project(marid_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)

find_package(tf2_ros REQUIRED)


# Install launch + config + imu dir (good)
install(
  DIRECTORY marid_controller launch config imu
  DESTINATION share/${PROJECT_NAME}
)

# Install the python node as an executable
install(PROGRAMS
  imu/imu.py
  marid_controller/marid_odom_pub.py
  marid_controller/marid_thrust_controller.py
  marid_controller/marid_ai_controller.py
  marid_controller/marid_safety_node.py
  marid_controller/marid_attitude_controller.py
  marid_controller/marid_data_logger.py
  marid_controller/marid_ai_guidance.py
  marid_controller/marid_guidance_tracker.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
