# Project name and cmake minimum requirement.
project(lltm)
cmake_minimum_required(VERSION 3.16)

# We find a LibTorch installation trough the torch package.
# This is the best approach if we want to make sure we are
# targetting the same LibTorch version as used by torch.
execute_process (
    COMMAND Rscript -e "cat(torch::torch_install_path())"
    OUTPUT_VARIABLE TORCH_HOME
)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${TORCH_HOME}")

# Now that the prefix path is set we can tell cmake to go
# and find Torch.
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Here we tell CMake what are the source files of our package.
# If you want to separate your implementation in multiple files
# add their paths after `src/lltm.cpp`, the spearator is a simple
# space.
set(LLTM_SRC src/lltm.cpp)

# On Windows we use module definition files to declare what are
# the exported functions from the library. It's similar to the
# Namespace file in the R ecossystem.
# We need to fill it manually as its used to link the Rcpp interface
# with the `csrc` library.
if(WIN32)
  set(LLRM_SRC ${LLTM_SRC} lltm.def)
endif()

# Tell cmake to build the shared library.
add_library(lltm SHARED ${LLTM_SRC})
add_library(lltm::library ALIAS lltm)

# Tell cmake what are the include files.
target_include_directories(lltm PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Tell cmake the libraries we want to link to.
target_link_libraries(lltm "${TORCH_LIBRARIES}")
set_property(TARGET lltm PROPERTY CXX_STANDARD 14)

# Syncronize the headers and the def file with the Rcpp
# interface.
add_custom_command(TARGET lltm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/src/lltm.def ${PROJECT_SOURCE_DIR}/../inst/def/lltm.def
    COMMENT "Copied def file to inst folder."
)

add_custom_command(TARGET lltm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/include/lltm/lltm.h ${PROJECT_SOURCE_DIR}/../src/lltm/lltm.h
    COMMENT "Copied lltm.h file to src folder."
)


# Set CPack related code to automatically generate installation bundles.
# The bundle name will have the same version as defined in the R DESCRIPTION
# file.
# The cpack configuration is used by the CI/CD workflows to create the pre-built
# binaries bundles and upload them to the GitHub Releases page.
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
