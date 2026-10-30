cmake_minimum_required(VERSION 3.14)
project(all-round LANGUAGES C CXX)
set(SPARSE4D_ARCH "x86" CACHE STRING "x86 or aarch64")
if(NOT SPARSE4D_ARCH MATCHES "^(x86|aarch64)$")
  message(FATAL_ERROR "SPARSE4D_ARCH must be x86 or aarch64")
endif()
set(DEPS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../deps_${SPARSE4D_ARCH}" CACHE PATH "SDK dependencies")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/deploy/sparse4d/${SPARSE4D_ARCH}" CACHE PATH "Install directory" FORCE)
endif()
include_directories(${DEPS_ROOT}/ucp/include ${DEPS_ROOT}/opencv/include ${DEPS_ROOT}/gflags/include ${DEPS_ROOT}/nlohmann/include ${CMAKE_CURRENT_SOURCE_DIR}/include/sparse4d)
link_directories(${DEPS_ROOT}/ucp/lib ${DEPS_ROOT}/opencv/lib ${DEPS_ROOT}/gflags/lib)
add_executable(sparse4d_J6P src/sparse4d/main.cc src/sparse4d/sparse4d_runtime.cc)
# ============================================================
# J6P AArch64 底层运行时动态库路径
# ============================================================

# ============================================================
# J6P AArch64 runtime libraries
# ============================================================

if(SPARSE4D_ARCH STREQUAL "aarch64")

    set(J6P_APPSDK_LIB_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../deps_aarch64/appsdk/appuser/lib"
    )

    target_link_options(sparse4d_J6P PRIVATE
        "-Wl,-rpath-link,${J6P_APPSDK_LIB_DIR}"
        "-Wl,--allow-shlib-undefined"
    )

endif()


target_link_libraries(sparse4d_J6P dnn hbucp gflags opencv_world pthread dl rt)
set_target_properties(sparse4d_J6P PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
install(TARGETS sparse4d_J6P DESTINATION bin)
install(DIRECTORY ${DEPS_ROOT}/ucp/lib/ DESTINATION lib USE_SOURCE_PERMISSIONS)
install(DIRECTORY ${DEPS_ROOT}/opencv/lib/ ${DEPS_ROOT}/gflags/lib/ DESTINATION lib FILES_MATCHING PATTERN "*.so*")
install(PROGRAMS run_sparse4d.sh DESTINATION bin)
option(BUILD_ALL_ROUND_SAMPLE "Build original board camera sample" OFF)
if(BUILD_ALL_ROUND_SAMPLE)
  include(cmake/OriginalAllRound.cmake)
endif()

option(SPARSE4D_BUILD_TESTS "Build tensor and decode tests" OFF)
if(SPARSE4D_BUILD_TESTS)
  enable_testing()
  add_executable(sparse4d_core_test tests/sparse4d/test_core.cc)
  add_executable(sparse4d_decode_fixture tests/sparse4d/decode_fixture.cc)
  add_test(NAME sparse4d_core COMMAND sparse4d_core_test)
endif()
