project(lltm)
cmake_minimum_required(VERSION 3.16)


# We find a LibTorch installation trough the torch package.
execute_process (
    COMMAND Rscript -e "cat(torch::torch_install_path())"
    OUTPUT_VARIABLE TORCH_HOME
)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${TORCH_HOME}")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(LLTM_SRC src/lltm.cpp)

add_library(lltm SHARED ${LLTM_SRC})
add_library(lltm::library ALIAS lltm)

target_include_directories(lltm PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(lltm "${TORCH_LIBRARIES}")
set_property(TARGET lltm PROPERTY CXX_STANDARD 14)


# If on WIndows we want to copy the .lib file to the package directory

if(WIN32)
  add_custom_command(TARGET lltm POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_LINKER_FILE:lltm> ${PROJECT_SOURCE_DIR}/../inst/lib/lltm.lib
      COMMENT "Copied lib file to inst folder."
  )
endif()

# Set CPack related code to automatically generate installation bundles.

set(CPACK_GENERATOR ZIP)
execute_process (
    COMMAND Rscript -e "cat(desc::description$new(file = '../../DESCRIPTION')$get('Version'))"
    OUTPUT_VARIABLE CPACK_PACKAGE_VERSION
)

if(DEFINED ${CUDA_VERSION_STRING})
  set(CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}+cu${CUDA_VERSION_STRING})
else()
  set(CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}+cpu)
endif()


install(TARGETS lltm DESTINATION lib)
include(CPack)
