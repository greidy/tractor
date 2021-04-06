include(CMakeFindDependencyMacro)
find_dependency(Glog)
find_dependency(Sophus)
find_dependency(apriltag)
find_dependency(Eigen3)
find_dependency(OpenCV)
find_dependency(farm_ng_core)

if (NOT TARGET farm_ng::farm_ng_perception)
  include("${CMAKE_CURRENT_LIST_DIR}/farm_ng_perceptionTargets.cmake")
endif()