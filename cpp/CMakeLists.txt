SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11" )
SET( CMAKE_INCLUDE_CURRENT_DIR ON)

#####################################
# Find dependent libraries
#####################################
# Find GTSAM components
find_package(GTSAM REQUIRED) # Uses installed package
include_directories(${GTSAM_INCLUDE_DIR})
set(GTSAM_LIBS gtsam)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${GTSAM_DIR}/../GTSAMCMakeTools")
include(GtsamBuildTypes)
include(GtsamTesting)

file(GLOB handical_srcs "*.cpp" "*.h")
add_library(handical ${handical_srcs})
target_link_libraries(handical ${Boost_LIBRARIES} ${GTSAM_LIBS})

set(test_libs ${GTSAM_LIBS})
set(excluded_test "")
gtsamAddTestsGlob(handical "tests/test*.cpp" "${excluded_test}" "${test_libs}")

# Wrap
include(GtsamCythonWrap)
include_directories(${GTSAM_EIGENCY_INSTALL_PATH})
configure_file(${PROJECT_SOURCE_DIR}/python/__init__.py
               ${PROJECT_BINARY_DIR}/cython/handical/__init__.py
               COPYONLY)
wrap_and_install_library_cython("../handical.h"
                                "from gtsam.gtsam cimport *"
                                "${CMAKE_INSTALL_PREFIX}/cython/handical"
                                "handical"
                                "handical")
install_cython_scripts("${PROJECT_SOURCE_DIR}/python/"
                       "${CMAKE_INSTALL_PREFIX}/cython/handical"
                       "*.py")
