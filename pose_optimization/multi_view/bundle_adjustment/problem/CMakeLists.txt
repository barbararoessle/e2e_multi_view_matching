add_library(problem
    include/ba_problem.h
    src/ba_problem.cpp
)
target_include_directories(problem PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(problem PUBLIC
    io
    ${CERES_LIBRARIES}
)
add_executable(test_problem
    test/test_ba_problem.cpp)
target_link_libraries(test_problem
    io
    problem
    Eigen3::Eigen
    ${CERES_LIBRARIES}
    ${GTEST_LIBRARIES}
    pthread
)
