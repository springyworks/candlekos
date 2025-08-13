# CMake generated Testfile for 
# Source directory: /home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main
# Build directory: /home/rustuser/projects/rust/from_github/candle/build/glslang-main
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(glslang-testsuite "bash" "runtests" "/home/rustuser/projects/rust/from_github/candle/build/glslang-main/localResults" "/home/rustuser/projects/rust/from_github/candle/build/glslang-main/StandAlone/glslang" "/home/rustuser/projects/rust/from_github/candle/build/glslang-main/StandAlone/spirv-remap")
set_tests_properties(glslang-testsuite PROPERTIES  WORKING_DIRECTORY "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/Test/" _BACKTRACE_TRIPLES "/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/CMakeLists.txt;331;add_test;/home/rustuser/projects/rust/from_github/candle/third_party/VkFFT/glslang-main/CMakeLists.txt;0;")
subdirs("External")
subdirs("glslang")
subdirs("OGLCompilersDLL")
subdirs("StandAlone")
subdirs("SPIRV")
subdirs("hlsl")
subdirs("gtests")
