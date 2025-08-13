# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main"
  "/home/rustuser/projects/rust/from_github/candle/build/_deps/glslang-main-build"
  "/home/rustuser/projects/rust/from_github/candle/build/_deps/glslang-main-subbuild/glslang-main-populate-prefix"
  "/home/rustuser/projects/rust/from_github/candle/build/_deps/glslang-main-subbuild/glslang-main-populate-prefix/tmp"
  "/home/rustuser/projects/rust/from_github/candle/build/_deps/glslang-main-subbuild/glslang-main-populate-prefix/src/glslang-main-populate-stamp"
  "/home/rustuser/projects/rust/from_github/candle/build/_deps/glslang-main-subbuild/glslang-main-populate-prefix/src"
  "/home/rustuser/projects/rust/from_github/candle/build/_deps/glslang-main-subbuild/glslang-main-populate-prefix/src/glslang-main-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/rustuser/projects/rust/from_github/candle/build/_deps/glslang-main-subbuild/glslang-main-populate-prefix/src/glslang-main-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/rustuser/projects/rust/from_github/candle/build/_deps/glslang-main-subbuild/glslang-main-populate-prefix/src/glslang-main-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
