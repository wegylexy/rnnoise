cmake_minimum_required(VERSION 3.6)
project(RnNoise LANGUAGES C)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(RN_NOISE_SRC
        src/rnnoise.c
        src/rnnoise_demo.c)

include_directories(include)
add_executable(rnnoise_demo ${RN_NOISE_SRC})
target_link_libraries(rnnoise_demo -lm)