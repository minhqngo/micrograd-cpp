cmake_minimum_required(VERSION 3.15)
project(micrograd_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen3)

include_directories(include)

set(SOURCES
    src/engine.cpp
    src/nn.cpp
    src/loss.cpp
    src/optimizer.cpp
    src/mnist_loader.cpp
)

add_library(micrograd STATIC ${SOURCES})
target_link_libraries(micrograd Eigen3::Eigen)
target_include_directories(micrograd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(train_mnist examples/train_mnist.cpp)
target_link_libraries(train_mnist micrograd Eigen3::Eigen)

add_executable(test_autograd tests/test_autograd.cpp)
target_link_libraries(test_autograd micrograd Eigen3::Eigen)

add_executable(test_training tests/test_training.cpp)
target_link_libraries(test_training micrograd Eigen3::Eigen)
