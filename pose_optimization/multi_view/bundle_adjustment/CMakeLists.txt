cmake_minimum_required (VERSION 3.11)

project(bundle_adjustment LANGUAGES CXX)

find_package(Boost COMPONENTS filesystem REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)
find_package(Theia REQUIRED)
find_package(GTest REQUIRED)

include_directories(${Boost_INCLUDE_DIR} ${GTEST_INCLUDE_DIRS})
include_directories(${THEIA_INCLUDE_DIRS})

add_executable(bundle_adjuster
    bundle_adjuster.cpp
)

target_link_libraries(bundle_adjuster PRIVATE
    io
    problem
    ${Boost_FILESYSTEM_LIBRARY}
    ${CERES_LIBRARIES}
)

add_executable(ba_initializer
    ba_initializer.cpp
)

target_link_libraries(ba_initializer PRIVATE
    io
    ba_init
    ${Boost_FILESYSTEM_LIBRARY}
)

add_subdirectory(io)
add_subdirectory(problem)
add_subdirectory(ba_init)
