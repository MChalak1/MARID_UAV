cmake_minimum_required(VERSION 3.8)
project(marid_localization)

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

include_directories(include)


install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  marid_localization/baro_to_alt.py
  marid_localization/imu_republisher.py
  marid_localization/imu_add_gravity.py
  marid_localization/camera_hud_overlay.py
  marid_localization/air_density_calculator.py
  marid_localization/airspeed_converter.py
  marid_localization/wind_estimator.py
  marid_localization/lidar_time_field.py
  marid_localization/optical_flow_estimator.py
  marid_localization/wheel_odometry.py
  marid_localization/forward_flow_estimator.py
  marid_localization/vx_error_monitor.py
  marid_localization/sun_sensor_sim.py
  DESTINATION lib/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  set(ament_cmake_copyright_FOUND TRUE)

  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
