cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(mandelbrot)

# FindCUDA module
find_package(CUDA REQUIRED)

# Set architecture
set(CUDA_ARCH "sm_86" CACHE STRING "CUDA architecture (default: sm_35)")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=${CUDA_ARCH})

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Find OpenGL libraries
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)
if (OPENGL_LIBRARIES)
    message("OpenGL libraries found: ${OPENGL_LIBRARIES}")
else ()
    message(FATAL_ERROR "OpenGL libraries not found")
endif ()

# Find GLUT library
find_package(GLUT REQUIRED)
if (GLUT_LIBRARIES)
    message("GLUT libraries found: ${GLUT_LIBRARIES}")
else ()
    message(FATAL_ERROR "GLUT libraries not found")
endif ()

# Include directories for OpenGL and GLUT
include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})

# List your CUDA source files
set(CUDA_SOURCES main.cpp mandelbrot.cu)

# Add the CUDA executable
cuda_add_executable(mandelbrot ${CUDA_SOURCES})

# Link OpenGL and GLUT libraries
target_link_libraries(mandelbrot ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

# Include directories for helper files
set(CUDA_SAMPLES_DIR "/usr/local/cuda/samples/")
file(TO_CMAKE_PATH ${CUDA_SAMPLES_DIR} EXPANDED_CUDA_SAMPLES_DIR)
target_include_directories(mandelbrot PRIVATE "${EXPANDED_CUDA_SAMPLES_DIR}/Common")
