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
if(WIN32)
  set(LLRM_SRC ${LLTM_SRC} lltm.def)
endif()

add_library(lltm SHARED ${LLTM_SRC})
add_library(lltm::library ALIAS lltm)

target_include_directories(lltm PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(lltm "${TORCH_LIBRARIES}")
set_property(TARGET lltm PROPERTY CXX_STANDARD 14)

add_custom_command(TARGET lltm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/src/lltm.def ${PROJECT_SOURCE_DIR}/../inst/def/lltm.def
    COMMENT "Copied def file to inst folder."
)

add_custom_command(TARGET lltm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/include/lltm/lltm.h ${PROJECT_SOURCE_DIR}/../src/lltm/lltm.h
    COMMENT "Copied lltm.h file to src folder."
)

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

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/../inst)
install(TARGETS lltm DESTINATION lib)
