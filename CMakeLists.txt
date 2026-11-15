cmake_minimum_required(VERSION 3.16)
project(homework_2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 生成 build/compile_commands.json，供 VS Code 的 IntelliSense 读取编译参数。
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV 4 REQUIRED
    COMPONENTS core imgproc imgcodecs highgui videoio aruco)
# 查找 OpenCV 4，并要求包含这几个模块。

add_executable(homework_2
    src/main.cpp
    src/Detector.cpp
)

target_include_directories(homework_2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
# 给 homework_2 这个目标添加头文件搜索路径。                
# ${CMAKE_CURRENT_SOURCE_DIR}/include：项目自己的 include/ 目录，这样 #include "Detector.hpp" 才能找到文件。
# ${OpenCV_INCLUDE_DIRS}：OpenCV 头文件所在目录，这样 #include <opencv2/...> 才能找到。

target_link_libraries(homework_2 PRIVATE ${OpenCV_LIBS})
# 把 OpenCV 的库链接进来。
# imread、detectMarkers 这些函数的实现不在源码里，在 OpenCV 的库文件里。

target_compile_options(homework_2 PRIVATE -Wall -Wextra -Wpedantic)
# 开启警告。
