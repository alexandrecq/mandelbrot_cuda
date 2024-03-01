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
# set(OpenGL_GL_PREFERENCE "LEGACY")
find_package(OpenGL REQUIRED)
if (OPENGL_LIBRARIES)
    message("OpenGL libraries found: ${OPENGL_LIBRARIES}")
else ()
    message(FATAL_ERROR "OpenGL libraries not found")
endif ()

# # Find GLUT library
# set(CMAKE_FIND_DEBUG_MODE TRUE)
# find_package(GLUT REQUIRED)
# if (GLUT_LIBRARY)
#     message("GLUT library found: ${GLUT_LIBRARY}")
# else ()
#     message(FATAL_ERROR "GLUT library not found")
# endif ()

# TODO Figure out why find_package fails to find this file:
set(GLUT_LIBRARY "/usr/lib/x86_64-linux-gnu/libglut.so")

# Include directories for OpenGL and GLUT
include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})

# List your CUDA source files
set(CUDA_SOURCES main.cpp mandelbrot.cu)

# Add the CUDA executable
cuda_add_executable(mandelbrot ${CUDA_SOURCES})

# Link OpenGL and GLUT libraries
target_link_libraries(mandelbrot ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})


# Include directories for helper files
set(CUDA_SAMPLES_DIR "~/software/cuda/cuda-samples")
file(TO_CMAKE_PATH ${CUDA_SAMPLES_DIR} EXPANDED_CUDA_SAMPLES_DIR)
target_include_directories(mandelbrot PRIVATE "${EXPANDED_CUDA_SAMPLES_DIR}/Common")
