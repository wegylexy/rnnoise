cmake_minimum_required(VERSION 3.6)
project(RnNoise LANGUAGES C)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(RN_NOISE_SRC
        src/_kiss_fft_guts.h
        src/arch.h
        src/celt_lpc.h
        src/common.h
        src/kiss_fft.h
        src/opus_types.h
        src/pitch.h
        src/rnn.h
        src/rnn_data.h
        src/tansig_table.h
        src/celt_lpc.c
        src/denoise.c
        src/kiss_fft.c
        src/pitch.c
        src/rnn.c
        src/rnn_data.c
        examples/rnnoise_demo.c)

include_directories(include)
add_executable(rnnoise_demo ${RN_NOISE_SRC})
target_link_libraries(rnnoise_demo -lm )