#
# Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.10)

project(PrebuildDonut)

# Variables
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/_bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_PRODUCTION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_INSTALL_PREFIX "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(DONUT_SHADERS_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/framework")
if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_ITERATOR_DEBUG_LEVEL=1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP")
endif()
set(PACKAGE_DIRECTORY "${CMAKE_SOURCE_DIR}/_package")

set(NVIGI_INSTALL_DIR  "${PACKAGE_DIRECTORY}/_bin" CACHE STRING "NVIGI Install Dir")

set (SHADERMAKE_BIN_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/_build/bin" CACHE STRING "")

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;Production")
set(CMAKE_EXE_LINKER_FLAGS_PRODUCTION ${CMAKE_EXE_LINKER_FLAGS_RELEASE})
set(CMAKE_CXX_FLAGS_PRODUCTION ${CMAKE_CXX_FLAGS_RELEASE})
set(CMAKE_C_FLAGS_PRODUCTION ${CMAKE_C_FLAGS_RELEASE})
set(ENV{VULKAN_SDK} ${VULKAN_SDK})

option (JSONCPP_WITH_PKGCONFIG_SUPPORT OFF)
option (JSONCPP_WITH_CMAKE_PACKAGE OFF)
# Add libraries
# option(DONUT_WITH_VULKAN "Force VK off in donut for the NVIGI sample" OFF)
add_subdirectory(Donut)

set(DONUT_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_build/donut" CACHE STRING "Donut lib Directory")
set(DONUT_3RD_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_build/donut/thirdparty" CACHE STRING "Donut 4rd-party lib Directory")
set(NVRHI_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_build/donut/nvrhi" CACHE STRING "NVRHI lib Directory")
set(DEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_package/" CACHE STRING "Dest Directory")

install(DIRECTORY "Donut/include/" DESTINATION "${DEST_DIR}/donut/include" FILES_MATCHING PATTERN "*.h")
install(DIRECTORY "Donut/nvrhi/include/" DESTINATION "${DEST_DIR}/donut/nvrhi/include" FILES_MATCHING PATTERN "*.h")
install(DIRECTORY "Donut/thirdparty/glfw/include/" DESTINATION "${DEST_DIR}/donut/thirdparty/glfw/include" FILES_MATCHING PATTERN "*.h")
install(DIRECTORY "Donut/thirdparty/imgui/" DESTINATION "${DEST_DIR}/donut/thirdparty/imgui" FILES_MATCHING PATTERN "*.h")
install(DIRECTORY "Donut/thirdparty/taskflow/taskflow/" DESTINATION "${DEST_DIR}/donut/thirdparty/taskflow/taskflow" FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY "Donut/nvrhi/thirdparty/DirectX-Headers/include/" DESTINATION "${DEST_DIR}/donut/nvrhi/thirdparty/DirectX-Headers/include" FILES_MATCHING PATTERN "*.h")
install(DIRECTORY "Donut/nvrhi/thirdparty/Vulkan-Headers/include/" DESTINATION "${DEST_DIR}/donut/nvrhi/thirdparty/Vulkan-Headers/include" FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

install(DIRECTORY "${DONUT_LIB_DIR}/$<CONFIG>/" DESTINATION "${DEST_DIR}/donut/$<CONFIG>")
install(DIRECTORY "${NVRHI_LIB_DIR}/$<CONFIG>/" DESTINATION "${DEST_DIR}/donut/nvrhi/$<CONFIG>")
install(DIRECTORY "${DONUT_3RD_LIB_DIR}/$<CONFIG>/" DESTINATION "${DEST_DIR}/donut/thirdparty/$<CONFIG>")
install(DIRECTORY "${DONUT_3RD_LIB_DIR}/glfw/src/$<CONFIG>/" DESTINATION "${DEST_DIR}/donut/thirdparty/glfw/src/$<CONFIG>")
install(DIRECTORY "${DONUT_3RD_LIB_DIR}/miniz/$<CONFIG>/" DESTINATION "${DEST_DIR}/donut/thirdparty/miniz/$<CONFIG>")
install(DIRECTORY "${DONUT_3RD_LIB_DIR}/jsoncpp/src/lib_json/$<CONFIG>/" DESTINATION "${DEST_DIR}/donut/thirdparty/jsoncpp/src/lib_json/$<CONFIG>")
install(DIRECTORY "${DONUT_LIB_DIR}/ShaderMake/$<CONFIG>/" DESTINATION "${DEST_DIR}/donut/ShaderMake/$<CONFIG>")

install(DIRECTORY "${DONUT_SHADERS_OUTPUT_DIR}/" DESTINATION "${DEST_DIR}/donut/shaders/framework")

