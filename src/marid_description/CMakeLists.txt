cmake_minimum_required(VERSION 3.8)
project(marid_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


find_package(ament_cmake REQUIRED)
find_package(gz-sim8 REQUIRED)
find_package(gz-common5 REQUIRED)
find_package(gz-math7 REQUIRED)
find_package(gz-plugin2 REQUIRED)

install(
  DIRECTORY meshes meshes_2 urdf launch rviz worlds
  DESTINATION share/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  set(ament_cmake_copyright_FOUND TRUE)

  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
