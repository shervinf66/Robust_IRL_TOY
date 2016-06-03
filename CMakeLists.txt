cmake_minimum_required(VERSION 2.8)
project(Robust_IRL_TOY)

set(CMAKE_BUILD_TYPE Debug)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

# Find LibXml2 library
# find_package(LibXml2 2.6.0 REQUIRED)

# Find Armadillo library
# find_package(Armadillo REQUIRED)

# The Google Glog_Pack library.
# Store the glog include directory in GLOG_INCLUDE_DIR.
# The PATHS variable may be specified to give hints for where to find logging.h

# finding boost
#find_package(Boost REQUIRED system serialization)

# Boost dependency
#set(BOOST_ROOT CACHE PATH /usr/include/boost)
find_package(Boost COMPONENTS mpi serialization REQUIRED)

find_path(GLOG_INCLUDE_DIR
    NAMES logging.h
    PATHS /usr/local/include/glog /usr/local/include/ /usr/include/glog/ /usr/include/
    )

if(GLOG_INCLUDE_DIR)
    message(STATUS "Found GLog include directory")
else()
    message(STATUS "ERROR: Could not find GLog include directory")
endif()

# Find libmlpack.so (or equivalent) and store it in MLPACK_LIBRARY.
# If this example script were smarter, it would also find other dependencies of mlpack and store them in
# an MLPACK_LIBRARIES variable instead.
find_library(GLOG_LIBRARY
    NAMES glog libglog
    PATHS /usr/lib64/ /usr/lib/ /usr/local/lib64/ /usr/local/
    )

if(GLOG_LIBRARY)
    message(STATUS "Found GLog link libraries")
else()
    message(STATUS "ERROR: Could not find GLog link libraries")
endif()

find_library(PTHREAD_LIBRARY
    NAMES pthread libpthread
    PATHS /usr/lib64/ /usr/lib/ /usr/local/lib64/ /usr/local/
    )

if(PTHREAD_LIBRARY)
    message(STATUS "Found PThread link libraries")
else()
    message(STATUS "ERROR: Could not find PThread link libraries")
endif()


# Define the framework files.
set(SRC
    src/main.cpp
    src/Sampler.cpp
    src/MCHMM.cpp
    src/DETree.cpp
    src/Sample.cpp
    src/Observation.cpp
    src/Timer.cpp
    src/data.cpp
    src/process.cpp
    src/RIRL.cpp
    )

set(HDRs
    include/Sampler.h
    include/MCHMM.h
    include/DETree.h
    include/Sample.h
    include/Observation.h
    include/Timer.h
    include/data.h
    include/process.h
    include/linear.h
    include/RIRL.h
    )

include_directories (
    include/
    ${GLOG_INCLUDE_DIR}
    ${LIBXML2_INCLUDE_DIR}
    ${ARMADILLO_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${THREADS_INCLUDE_DIR}
    )

add_executable(${PROJECT_NAME} ${SRC} ${HDRs})

target_link_libraries(${PROJECT_NAME} 
    ${GLOG_LIBRARY}
    ${LIBXML2_LIBRARIES}
    ${Boost_LIBRARIES}
    ${THREADS_LIBRARY}
    ${PTHREAD_LIBRARY}
    )

