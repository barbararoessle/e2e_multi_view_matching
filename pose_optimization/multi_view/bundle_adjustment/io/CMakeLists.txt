add_library(io
    include/file_utils.h
    src/file_utils.cpp
)
target_include_directories(io PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
