add_library(ba_init  
    include/ba_init.h
    src/ba_init.cpp
)
target_include_directories(ba_init PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(ba_init PUBLIC
    io
    Eigen3::Eigen
    ${CERES_LIBRARIES}
    ${THEIA_LIBRARIES}
)

add_executable(test_ba_init
    test/test_ba_init.cpp
)
target_link_libraries(test_ba_init
    ba_init
    Eigen3::Eigen
    ${CERES_LIBRARIES}
    ${THEIA_LIBRARIES}
    ${GTEST_LIBRARIES}
    pthread
)
