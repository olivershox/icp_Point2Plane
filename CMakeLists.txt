cmake_minimum_required(VERSION 3.5)
project(icp_point2plane)
set(DEFAULT_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-w")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g -ggdb ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-g ${CMAKE_CXX_FLAGS}")
find_package(PCL REQUIRED)



include_directories(${PROJECT_SOURCE_DIR}/sophus)
#find_package(Sophus REQUIRED)
include_directories(${Sophus_DIRECTORIES})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PCL_INCLUDE_DIRS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
add_library(kd src/kdtree.cpp
src/icp_3d.cpp)
target_link_libraries(kd
        ${PCL_LIBRARIES}
        ${GLOG_LIBRARIES}

        gflags
        )

add_executable(icp src/test_icp.cpp)
target_link_libraries(icp glog gflags kd tbb)
target_link_libraries(icp ${PCL_LIBRARIES})

#add_executable(icp_align src/icp_align.cpp)
#target_link_libraries(icp_align glog gflags )
#target_link_libraries(icp_align ${PCL_LIBRARIES})
