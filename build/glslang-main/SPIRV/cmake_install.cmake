# Install script for directory: /home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/SPIRV

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/llvm-objdump-19")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/rustuser/projects/rust/from_github/candle/build/glslang-main/SPIRV/libSPVRemapper.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/rustuser/projects/rust/from_github/candle/build/glslang-main/SPIRV/libSPIRV.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake" TYPE FILE FILES "/home/rustuser/projects/rust/from_github/candle/build/glslang-main/SPIRV/SPVRemapperTargets.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake" TYPE FILE FILES "/home/rustuser/projects/rust/from_github/candle/build/glslang-main/SPIRV/SPIRVTargets.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/glslang/SPIRV" TYPE FILE FILES
    "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/SPIRV/bitutils.h"
    "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/SPIRV/spirv.hpp"
    "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/SPIRV/GLSL.std.450.h"
    "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/SPIRV/GLSL.ext.EXT.h"
    "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/SPIRV/GLSL.ext.KHR.h"
    "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/SPIRV/GlslangToSpv.h"
    "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/SPIRV/hex_float.h"
    "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/SPIRV/Logger.h"
    "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/SPIRV/SpvBuilder.h"
    "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/SPIRV/spvIR.h"
    "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/SPIRV/doc.h"
    "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/SPIRV/SpvTools.h"
    "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/SPIRV/disassemble.h"
    "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/SPIRV/GLSL.ext.AMD.h"
    "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/SPIRV/GLSL.ext.NV.h"
    "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/SPIRV/GLSL.ext.ARM.h"
    "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/SPIRV/NonSemanticDebugPrintf.h"
    "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/SPIRV/NonSemanticShaderDebugInfo100.h"
    "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/SPIRV/SPVRemapper.h"
    "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/SPIRV/doc.h"
    )
endif()

