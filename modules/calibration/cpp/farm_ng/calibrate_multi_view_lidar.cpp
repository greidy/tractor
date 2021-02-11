#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>

#include <ceres/ceres.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <Eigen/Dense>
#include <boost/asio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "farm_ng/calibration/calibrate_multi_view_apriltag_rig.pb.h"
#include "farm_ng/calibration/calibrate_multi_view_lidar.pb.h"
#include "farm_ng/calibration/local_parameterization.h"
#include "farm_ng/calibration/multi_view_apriltag_rig_calibrator.h"
#include "farm_ng/calibration/multi_view_lidar_model.pb.h"

#include "farm_ng/perception/apriltag.h"
#include "farm_ng/perception/point_cloud.pb.h"

#include "farm_ng/core/blobstore.h"
#include "farm_ng/core/event_log_reader.h"
#include "farm_ng/core/init.h"
#include "farm_ng/core/ipc.h"

#include "farm_ng/perception/tensor.h"
#include "farm_ng/perception/time_series.h"

#include <Eigen/Dense>
DEFINE_bool(interactive, false, "receive program args via eventbus");
DEFINE_string(output_config, "", "Output the config to a file.");
DEFINE_string(config,
              "/blobstore/logs/capture_01292021/multi_view_lidar_config.json",
              "Load config from a file rather than args.");

DEFINE_string(name, "", "Name of calibration output.");
DEFINE_string(event_log, "", "Path to event log containing input data.");
DEFINE_string(
    calibrate_multi_view_apriltag_rig_result, "",
    "Path to result of calibrate_multi_view_apriltag_rig, containing camera "
    "rig and apriltag rig to calibrate the LIDARs with respect to.");

namespace fs = boost::filesystem;

namespace farm_ng::calibration {

void SavePly(std::string ply_path, const Eigen::MatrixXd& points) {
  std::ofstream out(ply_path);
  out << "ply\n";
  out << "format ascii 1.0\n";
  out << "element vertex " << points.cols() << "\n";
  out << "property float x\n";
  out << "property float y\n";
  out << "property float z\n";
  out << "end_header\n";
  for (int i = 0; i < points.cols(); ++i) {
    auto p = points.col(i);
    out << float(p.x()) << " " << float(p.y()) << " " << float(p.z()) << "\n";
  }
  out.close();
}

Eigen::Map<const Eigen::MatrixXd> PointCloudGetData(
    const perception::PointCloud& cloud, std::string name) {
  for (auto& point_data : cloud.point_data()) {
    CHECK_EQ(point_data.shape_size(), 2);
    if (point_data.shape(0).name() == name) {
      return perception::TensorToEigenMapXd(point_data);
    }
  }
  LOG(FATAL) << "No point_data named: " << name;
  return Eigen::Map<const Eigen::MatrixXd>(nullptr, 0, 0);
}

template <typename Scalar>
Eigen::Matrix<Scalar, 3, Eigen::Dynamic> PointsToSphericalCoordinates(
    const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& xyz) {
  Eigen::Matrix<Scalar, 3, Eigen::Dynamic> coords(3, xyz.cols());
  coords.row(2) = xyz.colwise().norm();
  const auto& r = coords.row(2);
  // theta
  coords.row(0) = xyz.row(2).cwiseQuotient(r.row(0)).array().acos();
  // phi
  coords.row(1) = (xyz.row(1).binaryExpr(xyz.row(0), [](Scalar y, Scalar x) {
    using std::atan2;
    return atan2(y, x);
  }));
  return coords;
}
template <typename Scalar>
Eigen::Matrix<Scalar, 3, Eigen::Dynamic> TransformPoints(
    const Sophus::SE3<Scalar>& a_pose_b,
    const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& xyz_b) {
  Eigen::Matrix<Scalar, 4, Eigen::Dynamic> h_xyz_b =
      Eigen::Matrix<Scalar, 4, Eigen::Dynamic>::Ones(4, xyz_b.cols());
  h_xyz_b.topRows(3) = xyz_b;
  return a_pose_b.matrix3x4() * h_xyz_b;
}

double JetToDouble(double a) { return a; }

template <typename T, int N>
double JetToDouble(const ceres::Jet<T, N>& jet) {
  return jet.a;
}

Eigen::MatrixXi MakeCoordMap(
    const Eigen::Matrix<double, 3, Eigen::Dynamic>& coords, double min_theta,
    double max_theta, double min_phi, double max_phi, int width, int height,
    int n_buckets) {
  Eigen::MatrixXi coord_map = -Eigen::MatrixXi::Ones(n_buckets, width * height);
  for (int i = 0; i < coords.cols(); ++i) {
    double theta = coords(0, i);
    double phi = coords(1, i);
    int theta_i = std::clamp(
        int(height * ((theta - min_theta) / (max_theta - min_theta)) + 0.5), 0,
        height - 1);
    int phi_i =
        std::clamp(int(width * ((phi - min_phi) / (max_phi - min_phi)) + 0.5),
                   0, width - 1);
    int map_col = phi_i + theta_i * width;
    for (int j = 0; j < coord_map.rows(); ++j) {
      if (coord_map(j, map_col) < 0) {
        coord_map(j, map_col) = i;
        break;
      }
    }
  }
  return coord_map;
}

std::vector<std::tuple<int, int, double>> SphericalCoordinateNearestNeighbors(
    const Eigen::Matrix3Xd& points_a, const Eigen::Matrix3Xd& points_b,
    int width, int height) {
  auto coords_a = PointsToSphericalCoordinates(points_a);
  const auto& theta = coords_a.row(0);
  const auto& phi = coords_a.row(1);
  const auto& r_a = coords_a.row(2);
  double min_theta = theta.minCoeff();
  double max_theta = theta.maxCoeff();
  double min_phi = phi.minCoeff();
  double max_phi = phi.maxCoeff();
  auto coords_b = PointsToSphericalCoordinates(points_b);
  auto coords_a_map = MakeCoordMap(coords_a, min_theta, max_theta, min_phi,
                                   max_phi, width, height, 10);
  auto coords_b_map = MakeCoordMap(coords_b, min_theta, max_theta, min_phi,
                                   max_phi, width, height, 10);
  CHECK_EQ(coords_a_map.cols(), coords_b_map.cols());
  std::vector<std::tuple<int, int, double>> neighbors;

  for (int i = 0; i < coords_a_map.cols(); ++i) {
    double min = std::numeric_limits<double>::max();
    int min_coord_a_i = -1;
    int min_coord_b_i = -1;
    for (int j = 0; j < coords_a_map.rows(); ++j) {
      int coord_a_i = coords_a_map(j, i);

      if (coord_a_i < 0) {
        break;
      }
      CHECK_LT(coord_a_i, points_a.cols());
      auto point_a_i = points_a.col(coord_a_i);
      for (int k = 0; k < coords_b_map.rows(); ++k) {
        int coord_b_i = coords_b_map(k, i);
        if (coord_b_i < 0) {
          break;
        }
        CHECK_LT(coord_b_i, points_b.cols());
        auto point_b_i = points_b.col(coord_b_i);
        double dist_i = (point_b_i - point_a_i).squaredNorm();
        if (dist_i < min && dist_i < 5.0) {
          min = dist_i;
          min_coord_a_i = coord_a_i;
          min_coord_b_i = coord_b_i;
        }
      }
    }
    if (min_coord_a_i >= 0) {
      neighbors.push_back({min_coord_a_i, min_coord_b_i, min});
    }
  }
  return neighbors;
}

// Finds point indices within +- x, +- y, +- z
std::tuple<std::vector<int>, Eigen::Vector3d> AxisAlignedBoxFilter(const Eigen::Matrix3Xd& points, double x,
                                      double y, double z) {
  std::vector<int> indices;
  Eigen::Vector3d m(0,0,0);
  for (int i = 0; i < points.cols(); ++i) {
    auto p_i = points.col(i);
    if (p_i.x() > -x && p_i.x() < x &&  // x
        p_i.y() > -y && p_i.y() < y &&  // y
        p_i.z() > -z && p_i.z() < z) {
       m += p_i.cwiseProduct(p_i);
      indices.push_back(i);
    }
  }
  if(indices.size() >0) {
    m = (m/indices.size()).array().sqrt();
  }
  return {indices, m};
}

std::vector<std::tuple<perception::ApriltagRig::Node, std::vector<int>, double>>
ApriltagRigPointCloudMatch(const perception::ApriltagRig& rig,
                           const Eigen::Matrix3Xd& points_rig,
                           double tolerance) {

  std::vector<std::tuple<perception::ApriltagRig::Node, std::vector<int>, double>>
      node_indices;
  for (auto node : rig.nodes()) {
    Sophus::SE3d tag_pose_rig =
        ProtoToSophus(node.pose(), node.frame_name(), rig.name());
    Eigen::Matrix3Xd points_tag = TransformPoints(tag_pose_rig, points_rig);
    // half the tag size + 1 pixel of border (note tag_size/8 is 1 apriltag
    // pixel)
    double half_tag = node.tag_size() / 8.0 + node.tag_size() / 2.0;

    auto [indices, error] =
        AxisAlignedBoxFilter(points_tag, half_tag, half_tag, tolerance);
    if (indices.empty()) {
      continue;
    }

    node_indices.push_back({node, indices, error.z()});
  }
  return node_indices;
}

Eigen::Matrix3Xd SelectPoints(const Eigen::MatrixXd& points,
                              const std::vector<int>& indices) {
  Eigen::Matrix3Xd out(3, indices.size());

  for (int i = 0; i < out.cols(); ++i) {
    CHECK_LT(indices[i], points.cols());
    CHECK_GT(indices[i], 0);
    out.col(i) = points.col(indices[i]);
  }
  return out;
}
std::vector<std::tuple<int, int, double>> BruteForceNeirestNeighbors(
    const Eigen::Matrix3Xd& points_a, const Eigen::Matrix3Xd& normals_a,
    const Eigen::Matrix3Xd& points_b, const double d_thresh) {
  CHECK_EQ(points_a.cols(), normals_a.cols());

  std::vector<std::tuple<int, int, double>> neighbors;
  neighbors.reserve(points_a.size());
  for (int i = 0; i < points_a.cols(); ++i) {
    Eigen::Vector3d p_a = points_a.col(i);
    Eigen::Vector3d n_a = normals_a.col(i);
    double min_d = std::numeric_limits<double>::max();
    int min_j = -1;
    Eigen::MatrixXd p_ab =
        (n_a.transpose() * (points_b.colwise() - p_a)).cwiseAbs();

    // Eigen::MatrixXd p_ab = ((points_b.colwise() -
    // p_a)).colwise().squaredNorm();

    CHECK_EQ(p_ab.rows(), 1)
        << p_ab.rows() << " x " << p_ab.cols() << " " << points_b.cols();
    CHECK_EQ(p_ab.cols(), points_b.cols())
        << p_ab.rows() << " x " << p_ab.cols();
    // for (int j = 0; j < points_b.cols(); ++j) {
    //   Eigen::Vector3d p_b = points_b.col(j);
    //   double d = std::abs(n_a.dot(p_b - p_a));
    //   CHECK_LT(std::abs(p_ab(0, j) - d), 1e-8);
    //   if (d < min_d) {
    //     min_j = j;
    //     min_d = d;
    //   }
    // }

    int min_j_fast = -1;
    int min_row_fast = -1;
    double min_d_fast = p_ab.minCoeff(&min_row_fast, &min_j_fast);
    // CHECK_EQ(min_j_fast, min_j);
    if (min_d_fast < d_thresh) {
      neighbors.push_back({i, min_j_fast, min_d_fast});
    }
    // if (min_d < d_thresh && min_j >= 0) {
    //   neighbors.push_back({i, min_j, min_d});
    // }
  }
  LOG(INFO) << "neighbors size: " << neighbors.size();
  return neighbors;
}

template <typename Scalar>
Eigen::Matrix<Scalar, 1, Eigen::Dynamic> MakeDepthMap(
    const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& coords, double min_theta,
    double max_theta, double min_phi, double max_phi, int width, int height) {
  Eigen::Matrix<Scalar, 1, Eigen::Dynamic> depth_map =
      Eigen::Matrix<Scalar, 1, Eigen::Dynamic>::Zero(1, width * height);
  Eigen::Matrix<Scalar, 1, Eigen::Dynamic> depth_map_count =
      Eigen::Matrix<Scalar, 1, Eigen::Dynamic>::Zero(1, width * height);
  const auto& r_a = coords.row(2);
  for (int i = 0; i < coords.cols(); ++i) {
    double theta = JetToDouble(coords(0, i));
    double phi = JetToDouble(coords(1, i));
    int theta_i = std::clamp(
        int(height * ((theta - min_theta) / (max_theta - min_theta)) + 0.5), 0,
        height - 1);
    int phi_i =
        std::clamp(int(width * ((phi - min_phi) / (max_phi - min_phi)) + 0.5),
                   0, width - 1);
    int depth_col = phi_i + theta_i * width;
    depth_map(0, depth_col) += r_a(0, i);
    depth_map_count(0, depth_col) += Scalar(1.0);
  }
  for (int i = 0; i < depth_map.cols(); ++i) {
    if (depth_map_count(0, i) > Scalar(0)) {
      depth_map(0, i) = depth_map(0, i) / depth_map_count(0, i);
    }
  }
  return depth_map;
}

struct PointCloudPlaneCostFunctor {
  PointCloudPlaneCostFunctor(
      const Eigen::Matrix3Xd& points_a, const Eigen::Matrix3Xd& normals_a,
      const Eigen::Matrix3Xd& points_b,
      const std::vector<std::tuple<int, int, double>>& matches)
      : points_a_(3, matches.size()),
        points_b_(3, matches.size()),
        normals_a_(3, matches.size()),
        matches_(matches) {
    CHECK_EQ(normals_a.cols(), points_a.cols());
    CHECK_EQ(normals_a.rows(), 3);
    for (size_t i = 0; i < matches.size(); ++i) {
      auto [first, second, score] = matches[i];
      points_a_.col(i) = points_a.col(first);
      normals_a_.col(i) = normals_a.col(first);
      points_b_.col(i) = points_b.col(second);
    }
  }

  template <class T>
  bool operator()(T const* const raw_a_pose_b, T* raw_residuals) const {
    Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>> residuals(raw_residuals, 1,
                                                              points_a_.cols());
    Eigen::Map<const Sophus::SE3<T>> a_pose_b(raw_a_pose_b);
    residuals = (normals_a_.cast<T>().array() *
                 (TransformPoints<T>(a_pose_b, points_b_.cast<T>()) -
                  points_a_.cast<T>())
                     .array())
                    .colwise()
                    .sum();

    return true;
  }
  Eigen::Matrix3Xd points_a_;
  Eigen::Matrix3Xd points_b_;
  Eigen::Matrix3Xd normals_a_;

  std::vector<std::tuple<int, int, double>> matches_;
};

struct PointPlaneCostFunctor {
  PointPlaneCostFunctor(const Eigen::Vector3d& point_a,
                        const Eigen::Vector3d& normal_a,
                        const Eigen::Matrix3Xd& points_b,
                        const perception::SE3Map& a_pose_b_map)
      : point_a_(point_a),
        normal_a_(normal_a),
        points_b_(points_b),
        a_pose_b_map_(a_pose_b_map) {}

  template <class T>
  bool operator()(T const* const raw_a_pose_b, T* raw_residuals) const {
    Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>> residuals(raw_residuals, 1,
                                                              points_b_.cols());
    auto a_pose_b = a_pose_b_map_.Map(raw_a_pose_b);
    residuals = (normal_a_.cast<T>().transpose() *
                 (TransformPoints<T>(a_pose_b, points_b_.cast<T>()).colwise() -
                  point_a_.cast<T>()));
    return true;
  }
  Eigen::Vector3d point_a_;
  Eigen::Vector3d normal_a_;
  Eigen::Matrix3Xd points_b_;
  perception::SE3Map a_pose_b_map_;
};

std::tuple<Sophus::SE3d, double> EstimateMotion(
    const perception::PointCloud& cloud_a,
    const perception::PointCloud& cloud_b, Sophus::SE3d a_pose_b,
    double d_thresh) {
  constexpr int width = 360 / 4;
  constexpr int height = 180 / 4;
  Eigen::Matrix3Xd points_a = PointCloudGetData(cloud_a, "xyz");
  Eigen::Matrix3Xd normals_a = PointCloudGetData(cloud_a, "nxyz");
  Eigen::Matrix3Xd points_b = PointCloudGetData(cloud_b, "xyz");

  auto begin_match = core::MakeTimestampNow();

  auto matches = BruteForceNeirestNeighbors(
      points_a, normals_a, TransformPoints<double>(a_pose_b, points_b),
      d_thresh);
  auto end_match = core::MakeTimestampNow();
  LOG(INFO) << "matching time (milliseconds) : "
            << google::protobuf::util::TimeUtil::DurationToMilliseconds(
                   end_match - begin_match);
  if (matches.size() < 20) {
    LOG(INFO) << "Not enough matches: " << matches.size();
    return {a_pose_b, -1.0};
  }

  ceres::Problem problem;
  problem.AddParameterBlock(a_pose_b.data(), Sophus::SE3d::num_parameters,
                            new LocalParameterizationSE3);

  ceres::CostFunction* cost_function1 = new ceres::AutoDiffCostFunction<
      PointCloudPlaneCostFunctor, ceres::DYNAMIC, Sophus::SE3d::num_parameters>(
      new PointCloudPlaneCostFunctor(points_a, normals_a, points_b, matches),
      matches.size());
  problem.AddResidualBlock(cost_function1, new ceres::CauchyLoss(d_thresh),
                           a_pose_b.data());

  // Set solver options (precision / method)
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.gradient_tolerance = 1e-8;
  options.function_tolerance = 1e-8;
  options.parameter_tolerance = 1e-8;
  options.max_num_iterations = 2000;

  // Solve
  ceres::Solver::Summary summary;
  // options.logging_type = ceres::PER_MINIMIZER_ITERATION;
  options.minimizer_progress_to_stdout = false;
  ceres::Solve(options, &problem, &summary);
  // LOG(INFO) << summary.FullReport() << std::endl;

  return {a_pose_b, summary.final_cost};
}

MultiViewLidarModel HackInitialCad(MultiViewLidarModel model) {
  Eigen::Matrix4d right_cad;
  right_cad <<  // initializer list
      0.342017940083864,
      -0.175192752934261, 0.923217865934204, -0.00682305750428412,  // row1
      -0.939692620785908, -0.062480906526118, 0.336264649881884,
      -0.00915281044501381,  // row2
      -0.00122764054260543, -0.982549558871709, -0.185996928092129,
      0.210803579561015,  // row3
      0.0, 0.0, 0.0, 1.0;

  // left_lidar_pose_left_camera =
  //   Sophus::SE3d::rotZ(-M_PI / 2) * left_lidar_pose_left_camera;

  Eigen::Matrix4d left_cad;
  left_cad <<  // initializer list
      0.342017940083859,
      0.175192752934267, -0.923217865934204,
      0.00726017882119311,  // row0
      0.93969262078591, -0.0624809065261251, 0.336264649881878,
      -0.00915284599211702,  // row1
      0.00122764054259999, -0.982549558871707, -0.185996928092137,
      0.211046652781509,  // row2
      0, 0, 0, 1;

  Sophus::SE3d left_lidar_pose_right_camera =
      Sophus::SE3d::rotZ(-M_PI / 2) * Sophus::SE3d(left_cad);
  Sophus::SE3d right_lidar_pose_left_camera =
      Sophus::SE3d::rotZ(-M_PI / 2) * Sophus::SE3d(right_cad);

  perception::PoseGraph posegraph;

  for (int i = 0; i < model.measurements_size() && i < 1; ++i) {
    const auto& m_i0 = model.measurements(i);
    CHECK_EQ(m_i0.camera_rig_pose_apriltag_rig().frame_a(), "flir_rig");
    auto left_cloud = m_i0.multi_view_pointclouds().point_clouds_per_view(0);
    auto right_cloud = m_i0.multi_view_pointclouds().point_clouds_per_view(1);
    posegraph.AddPose(left_cloud.frame_name(), "right_fork_flir_camera",
                      left_lidar_pose_right_camera);
    posegraph.AddPose(right_cloud.frame_name(), "left_fork_flir_camera",
                      right_lidar_pose_left_camera);
  }
  model.clear_measurements();
  model.mutable_lidar_poses()->CopyFrom(posegraph.ToNamedSE3Poses());
  farm_ng::core::WriteProtobufToJsonFile("/blobstore/scratch/model.json",
                                         model);
  return model;
}
void SavePlyFilesInTagRig(MultiViewLidarModel model, std::string prefix) {
  perception::PoseGraph posegraph;
  posegraph.AddPoses(model.camera_rig().camera_pose_rig());
  posegraph.AddPoses(model.lidar_poses());
  posegraph = posegraph.AveragePoseGraph(model.camera_rig().name());
  std::ofstream report_csv(prefix +"_report.csv");
  report_csv  << "# frame_number, point_coud, tag_id, error";
  for (int i = 0; i < model.measurements_size(); ++i) {
    const auto& m_i0 = model.measurements(i);
    Sophus::SE3d camera_rig_pose_apriltag_rig =
        ProtoToSophus(m_i0.camera_rig_pose_apriltag_rig(),
                      model.camera_rig().name(), model.apriltag_rig().name());
    for (const auto& cloud :
         m_i0.multi_view_pointclouds().point_clouds_per_view()) {
      Sophus::SE3d cloud_pose_camera_rig = posegraph.CheckAverageAPoseB(
          cloud.frame_name(), model.camera_rig().name());
      Sophus::SE3d cloud_pose_apriltag_rig =
          cloud_pose_camera_rig * camera_rig_pose_apriltag_rig;
      auto points_cloud = PointCloudGetData(cloud, "xyz");
      auto points_apriltag_rig = TransformPoints<double>(
          cloud_pose_apriltag_rig.inverse(), points_cloud);
      SavePly(prefix + cloud.frame_name() + std::to_string(i) +
                  ".ply",
              points_apriltag_rig);
      auto apriltag_rig_matches = ApriltagRigPointCloudMatch(
          model.apriltag_rig(), points_apriltag_rig, 0.1);
      for (auto [node, indices, error] : apriltag_rig_matches) {
        LOG(INFO) << node.frame_name() << "<->" << cloud.frame_name()
                  << "  matches: " << indices.size() << " error: " << error;
        report_csv  << i << ", "  << cloud.frame_name() <<  "," <<  node.id() << ", " << error << std::endl;
        SavePly(prefix + std::to_string(node.id()) +
                    cloud.frame_name() + std::to_string(i) + ".ply",
                SelectPoints(points_apriltag_rig, indices));
      }
    }
    }
}

MultiViewLidarModel Solve(MultiViewLidarModel model) {
  perception::PoseGraph posegraph;
  posegraph.AddPoses(model.camera_rig().camera_pose_rig());
  posegraph.AddPoses(model.lidar_poses());
  posegraph = posegraph.AveragePoseGraph(model.camera_rig().name());
  ceres::Problem problem;

  for (auto pose : posegraph.MutablePoseEdges()) {
    problem.AddParameterBlock(pose->GetAPoseB().data(),
                              Sophus::SE3d::num_parameters,
                              new LocalParameterizationSE3);
  }
  for (auto camera_pose : model.camera_rig().camera_pose_rig()) {
    problem.SetParameterBlockConstant(
        posegraph.MutablePoseEdge(camera_pose.frame_a(), camera_pose.frame_b())
            ->GetAPoseB()
            .data());
  }

  std::map<int, ApriltagRigTagStats> tag_stats;
  std::map<int, std::vector<std::tuple<PerImageRmse, ceres::ResidualBlockId, int>>>
      tag_id_to_per_image_rmse_block_id;

  for (int i = 0; i < model.measurements_size(); ++i) {
    const auto& m_i0 = model.measurements(i);
    Sophus::SE3d camera_rig_pose_apriltag_rig =
        ProtoToSophus(m_i0.camera_rig_pose_apriltag_rig(),
                      model.camera_rig().name(), model.apriltag_rig().name());
    for (const auto& cloud :
         m_i0.multi_view_pointclouds().point_clouds_per_view()) {
      auto* cloud_rig_edge = posegraph.MutablePoseEdge(
          cloud.frame_name(), model.camera_rig().name());
      Sophus::SE3d cloud_pose_camera_rig = cloud_rig_edge->GetAPoseBMapped(
          cloud.frame_name(), model.camera_rig().name());
      Sophus::SE3d cloud_pose_apriltag_rig =
          cloud_pose_camera_rig * camera_rig_pose_apriltag_rig;
      auto points_cloud = PointCloudGetData(cloud, "xyz");
      auto points_apriltag_rig = TransformPoints<double>(
          cloud_pose_apriltag_rig.inverse(), points_cloud);
      auto apriltag_rig_matches = ApriltagRigPointCloudMatch(
          model.apriltag_rig(), points_apriltag_rig, 0.1);
      for (auto [node, indices, error] : apriltag_rig_matches) {
        LOG(INFO) << node.frame_name() << "<->" << cloud.frame_name()
                  << "  matches: " << indices.size() << " error: " << error;
        //        SavePly("/blobstore/scratch/tag_" + std::to_string(node.id())
        //        +
        //"_left_" + std::to_string(i) + ".ply",
        // SelectPoints(points_rig, indices));
        Sophus::SE3d apriltag_rig_pose_tag = ProtoToSophus(
            node.pose(), model.apriltag_rig().name(), node.frame_name());
        Sophus::SE3d camera_rig_pose_tag =
            camera_rig_pose_apriltag_rig * apriltag_rig_pose_tag;
        // tag normal
        Eigen::Vector3d normal_camera_rig =
            camera_rig_pose_tag.rotationMatrix().col(2);
        Eigen::Vector3d point_camera_rig = camera_rig_pose_tag.translation();

        auto tag_points_cloud = SelectPoints(points_cloud, indices);
        ceres::CostFunction* cost_function1 =
            new ceres::AutoDiffCostFunction<PointPlaneCostFunctor,
                                            ceres::DYNAMIC,
                                            Sophus::SE3d::num_parameters>(
                new PointPlaneCostFunctor(
                    point_camera_rig, normal_camera_rig, tag_points_cloud,
                    cloud_rig_edge->GetAPoseBMap(model.camera_rig().name(),
                                                 cloud.frame_name())),
                tag_points_cloud.cols());
        auto block_id = problem.AddResidualBlock(
            cost_function1, new ceres::CauchyLoss(0.05),
            cloud_rig_edge->GetAPoseB().data());

      ApriltagRigTagStats& stats = tag_stats[node.id()];
      stats.set_tag_id(node.id());
      stats.set_n_frames(stats.n_frames() + 1);

        PerImageRmse per_image_rmse;
        per_image_rmse.set_frame_number(i);
        per_image_rmse.set_camera_name(cloud.frame_name());

        tag_id_to_per_image_rmse_block_id[node.id()].push_back(
            std::make_tuple(per_image_rmse, block_id, tag_points_cloud.cols()));
      }
    }
  }

  // Set solver options (precision / method)
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.gradient_tolerance = 1e-20;
  options.function_tolerance = 1e-20;
  options.parameter_tolerance = 1e-20;
  options.max_num_iterations = 2000;

  // Solve
  ceres::Solver::Summary summary;
  options.logging_type = ceres::PER_MINIMIZER_ITERATION;
  options.minimizer_progress_to_stdout = false;
  ceres::Solve(options, &problem, &summary);
  LOG(INFO) << summary.FullReport() << std::endl;


  double total_rmse = 0.0;
  double total_count = 0;
  std::vector<ApriltagRigTagStats> out_tag_stats;
  model.clear_tag_stats();
  for (auto it : tag_id_to_per_image_rmse_block_id) {
    int tag_id = it.first;
    auto& stats = tag_stats[tag_id];
    for (auto per_image_rmse_block : it.second) {
      auto [image_rmse, block, n_residuals] = per_image_rmse_block;
      double cost;
      Eigen::Matrix<double, 1, Eigen::Dynamic> residuals(1, n_residuals);
      problem.EvaluateResidualBlockAssumingParametersUnchanged(
          block, false, &cost, residuals.data(), nullptr);

      double squared_mean = residuals.squaredNorm()/residuals.cols();

      total_rmse += squared_mean;
      total_count += 1;

      stats.set_tag_rig_rmse(stats.tag_rig_rmse() + squared_mean);

      image_rmse.set_rmse(std::sqrt(squared_mean));
      stats.add_per_image_rmse()->CopyFrom(image_rmse);

    }
    stats.set_tag_rig_rmse(std::sqrt(stats.tag_rig_rmse() / stats.n_frames()));
    model.add_tag_stats()->CopyFrom(stats);
  }
  double rmse = std::sqrt(total_rmse / total_count);
  LOG(INFO) << "model rmse (meters): " << rmse << "\n";
  model.set_rmse(rmse);
  model.mutable_lidar_poses()->CopyFrom(posegraph.ToNamedSE3Poses());
  return model;
}

void PointCloudToDepthMap(const perception::PointCloud& cloud_a,
                          std::string frame_name) {
  auto xyz_b = PointCloudGetData(cloud_a, "xyz");
  auto xyz_a = TransformPoints<double>(Sophus::SE3d::rotZ(0.0), xyz_b);
  auto coords_a = PointsToSphericalCoordinates(xyz_a);

  const auto& theta = coords_a.row(0);
  const auto& phi = coords_a.row(1);
  const auto& r_a = coords_a.row(2);

  auto max_depth = r_a.maxCoeff();
  auto min_theta = theta.minCoeff();
  auto max_theta = theta.maxCoeff();
  auto min_phi = phi.minCoeff();
  auto max_phi = phi.maxCoeff();

  auto max_x = std::min(10.0, xyz_a.row(0).maxCoeff());
  auto min_x = std::max(-10.0, xyz_a.row(0).minCoeff());

  auto max_y = std::min(10.0, xyz_a.row(1).maxCoeff());
  auto min_y = std::max(-10.0, xyz_a.row(1).minCoeff());

  auto max_z = std::min(5.0, xyz_a.row(2).maxCoeff());
  auto min_z = std::max(-3.0, xyz_a.row(2).minCoeff());

  LOG(INFO) << "theta= " << theta.rows() << " x " << theta.cols()
            << " min_theta= " << min_theta << " max_theta= " << max_theta;
  LOG(INFO) << "phi= " << phi.rows() << " x " << phi.cols()
            << " min_phi= " << min_phi << " max_phi: " << max_phi;

  int map_w = 360;
  int map_h = 180;
  Eigen::MatrixXd xyz_map = Eigen::MatrixXd::Zero(3, map_w * map_h);
  Eigen::MatrixXd depth_map = Eigen::MatrixXd::Zero(1, map_w * map_h);

  for (int i = 0; i < theta.cols(); ++i) {
    // int theta_i =
    // std::clamp(int(map_h * (theta(0, i) / (2 * M_PI))), 0, map_h - 1);
    int theta_i = std::clamp(
        int(map_h * ((theta(0, i) - min_theta) / (max_theta - min_theta)) +
            0.5),
        0, map_h - 1);
    int phi_i = std::clamp(int(map_w * (M_PI - phi(0, i)) / (2 * M_PI) + 0.5),
                           0, map_w - 1);
    auto p = xyz_a.col(i);

    xyz_map.col(phi_i + theta_i * map_w) = Eigen::Vector3d(
        (p.x() - min_x) / (max_x - min_x), (p.y() - min_y) / (max_y - min_y),
        (p.z() - min_z) / (max_z - min_z));

    depth_map(0, phi_i + theta_i * map_w) = r_a(0, i) / max_depth;
  }
  static int out = 0;
  cv::Mat out_depth;
  cv::Mat out_depth_shallow(map_h, map_w, CV_64FC1, depth_map.data());

  cv::dilate(out_depth_shallow, out_depth_shallow, cv::Mat(), cv::Point(-1, -1),
             5);

  out_depth_shallow.convertTo(out_depth, CV_16UC1,
                              std::numeric_limits<uint16_t>::max() - 1);

  cv::Mat out_xyz;
  cv::Mat(map_h, map_w, CV_64FC3, xyz_map.data())
      .convertTo(out_xyz, CV_16UC3, std::numeric_limits<uint16_t>::max() - 1);

  std::string out_xyz_path = "/blobstore/scratch/depth_" + frame_name +
                             std::to_string(out) + ".xyz.png";

  // CHECK(cv::imwrite(out_xyz_path, out_xyz))
  //     << "Didn't write to : " << out_xyz_path;

  std::string out_path =
      "/blobstore/scratch/depth_" + frame_name + std::to_string(out++) + ".png";

  CHECK(cv::imwrite(out_path, out_depth)) << "Didn't write to : " << out_path;
}

void PointCloudAlign(const perception::PointCloud& cloud_a,
                     const perception::PointCloud& cloud_b) {
  auto xyz_a = PointCloudGetData(cloud_a, "xyz");
  auto nxyz_a = PointCloudGetData(cloud_a, "nxyz");
  auto xyz_b = PointCloudGetData(cloud_b, "xyz");
  auto nxyz_b = PointCloudGetData(cloud_b, "nxyz");

  auto r_a = xyz_a.colwise().norm();
  LOG(INFO) << "r_a= " << r_a.rows() << " x " << r_a.cols();
  auto z_a = xyz_a.row(2);
  auto y_a = xyz_a.row(1);
  auto x_a = xyz_a.row(0);
  using std::acos;
  using std::atan;

  auto theta = xyz_a.row(2).cwiseQuotient(r_a.row(0)).array().acos();
  auto phi = (xyz_a.row(1).cwiseQuotient(xyz_a.row(0))).array().atan();
}

class CalibrateMultiViewLidarProgram {
 public:
  CalibrateMultiViewLidarProgram(
      core::EventBus& bus, CalibrateMultiViewLidarConfiguration configuration,
      bool interactive)
      : bus_(bus), timer_(bus_.get_io_service()) {
    if (interactive) {
      status_.mutable_input_required_configuration()->CopyFrom(configuration);
    } else {
      set_configuration(configuration);
    }
    bus_.AddSubscriptions({"^" + bus_.GetName() + "/"});
    bus_.GetEventSignal()->connect(
        std::bind(&CalibrateMultiViewLidarProgram::on_event, this,
                  std::placeholders::_1));
    on_timer(boost::system::error_code());
  }

  int run() {
    while (status_.has_input_required_configuration()) {
      bus_.get_io_service().run_one();
    }
    LOG(INFO) << "config:\n" << configuration_.DebugString();

    CHECK(configuration_.has_event_log()) << "Please specify an event log.";

    MultiViewLidarModel model;
    model.mutable_lidar_poses()->CopyFrom(configuration_.lidar_poses());

    auto mv_rig_result = core::ReadProtobufFromResource<
        calibration::CalibrateMultiViewApriltagRigResult>(
        configuration_.calibrate_multi_view_apriltag_rig_result());

    LOG(INFO) << mv_rig_result.DebugString();
    model.mutable_camera_rig()->CopyFrom(
        core::ReadProtobufFromResource<perception::MultiViewCameraRig>(
            mv_rig_result.camera_rig_solved()));
    const auto& camera_rig = model.camera_rig();
    model.mutable_apriltag_rig()->CopyFrom(
        core::ReadProtobufFromResource<perception::ApriltagRig>(
            mv_rig_result.apriltag_rig_solved()));
    const auto& apriltag_rig = model.apriltag_rig();
    std::map<std::string, perception::TimeSeries<core::Event>> event_series;

    std::set<std::string> lidar_names;
    if (configuration_.include_lidars_size() > 0) {
      for (auto lidar_name : configuration_.include_lidars()) {
        lidar_names.insert(lidar_name);
      }
    }
    core::EventLogReader log_reader(configuration_.event_log());
    while (true) {
      core::Event event;
      try {
        event = log_reader.ReadNext();
      } catch (const std::runtime_error& e) {
        break;
      }

      if (event.data().Is<perception::PointCloud>()) {
        if (configuration_.include_lidars_size() == 0) {
          lidar_names.insert(event.name());
        }
      }
      event_series[event.name()].insert(event);
    }
    CHECK_GT(lidar_names.size(), 0);

    auto time_window =
        google::protobuf::util::TimeUtil::MillisecondsToDuration(200);

    perception::ApriltagsFilter tag_filter;
    std::string root_camera_name = camera_rig.root_camera_name();

    int steady_count = 2;

    for (auto event : event_series[root_camera_name + "/apritags"]) {
      perception::ApriltagDetections detections;
      CHECK(event.data().UnpackTo(&detections));
      if (!tag_filter.AddApriltags(detections, steady_count, 7)) {
        continue;
      }

      MultiViewLidarModel::Measurement measurement;
      for (auto lidar_name : lidar_names) {
        auto closest_event =
            event_series[lidar_name].FindNearest(event.stamp(), time_window);

        CHECK(closest_event && closest_event->stamp() == event.stamp())
            << closest_event->name() << " "
            << closest_event->stamp().ShortDebugString()
            << " != " << event.name() << " "
            << event.stamp().ShortDebugString();
        auto* cloud = measurement.mutable_multi_view_pointclouds()
                          ->add_point_clouds_per_view();
        CHECK(closest_event->data().UnpackTo(cloud))

            << closest_event->name() << " is not a PointCloud.";
        PointCloudToDepthMap(*cloud, closest_event->name());
      }
      auto* mv_detections = measurement.mutable_multi_view_detections();
      for (auto camera : camera_rig.cameras()) {
        // HACK
        std::string apriltags_topic = camera.frame_name() + "/apritags";
        CHECK(event_series.count(apriltags_topic) > 0)
            << apriltags_topic << " not in event log";
        auto closest_event = event_series[apriltags_topic].FindNearest(
            event.stamp(), time_window);
        // HACK we only have 3 images in current dataset per point cloud.
        CHECK(closest_event && closest_event->stamp() == event.stamp())
            << closest_event->name() << " "
            << closest_event->stamp().ShortDebugString()
            << " != " << event.name() << " "
            << event.stamp().ShortDebugString();
        CHECK(closest_event->data().UnpackTo(
            mv_detections->add_detections_per_view()))
            << closest_event->name() << " is not an ApriltagDetections.";
      }
      // std::optional<std::tuple<perception::NamedSE3Pose, double,
      //                         std::vector<ApriltagRigTagStats>>>
      auto camera_rig_pose_est = EstimateMultiViewCameraRigPoseApriltagRig(
          camera_rig, apriltag_rig, *mv_detections);
      if (!camera_rig_pose_est) {
        LOG(WARNING) << "Couldn't estimate camera pose for frame: "
                     << event.name() << " " << event.stamp().ShortDebugString();
        continue;
      }
      auto [pose, rmse, stats] = *camera_rig_pose_est;
      measurement.mutable_camera_rig_pose_apriltag_rig()->CopyFrom(pose);
      model.add_measurements()->CopyFrom(measurement);
    }
    SavePlyFilesInTagRig(model, "/blobstore/scratch/init_");
    model = Solve(model);
    SavePlyFilesInTagRig(model, "/blobstore/scratch/solved_");

    LOG(INFO) << "Measurements: " << model.measurements_size();
    CalibrateMultiViewLidarResult result;

    send_status();
    return 0;
  }

  void send_status() {
    bus_.Send(core::MakeEvent(bus_.GetName() + "/status", status_));
  }

  void on_timer(const boost::system::error_code& error) {
    if (error) {
      LOG(WARNING) << "timer error: " << __PRETTY_FUNCTION__ << error;
      return;
    }
    timer_.expires_from_now(boost::posix_time::millisec(1000));
    timer_.async_wait(std::bind(&CalibrateMultiViewLidarProgram::on_timer, this,
                                std::placeholders::_1));

    send_status();
  }

  bool on_configuration(const core::Event& event) {
    CalibrateMultiViewLidarConfiguration configuration;
    if (!event.data().UnpackTo(&configuration)) {
      return false;
    }
    LOG(INFO) << configuration.ShortDebugString();
    set_configuration(configuration);
    return true;
  }

  void set_configuration(CalibrateMultiViewLidarConfiguration configuration) {
    configuration_ = configuration;
    status_.clear_input_required_configuration();
    send_status();
  }

  void on_event(const core::Event& event) {
    CHECK(event.name().rfind(bus_.GetName() + "/", 0) == 0);
    if (on_configuration(event)) {
      return;
    }
  }

 private:
  core::EventBus& bus_;
  boost::asio::deadline_timer timer_;
  CalibrateMultiViewLidarConfiguration configuration_;
  CalibrateMultiViewLidarStatus status_;
  CalibrateMultiViewLidarResult result_;
};

}  // namespace farm_ng::calibration

void Cleanup(farm_ng::core::EventBus& bus) {}

int Main(farm_ng::core::EventBus& bus) {
  farm_ng::calibration::CalibrateMultiViewLidarConfiguration config;
  if (!FLAGS_config.empty()) {
    config = farm_ng::core::ReadProtobufFromJsonFile<
        farm_ng::calibration::CalibrateMultiViewLidarConfiguration>(
        FLAGS_config);
    farm_ng::calibration::CalibrateMultiViewLidarProgram program(
        bus, config, FLAGS_interactive);
    return program.run();
  } else {
    config.set_name(FLAGS_name);
    config.mutable_event_log()->CopyFrom(
        farm_ng::core::EventLogResource(FLAGS_event_log));
    config.mutable_calibrate_multi_view_apriltag_rig_result()->CopyFrom(
        farm_ng::core::ProtobufJsonResource<
            farm_ng::calibration::CalibrateMultiViewApriltagRigResult>(
            FLAGS_calibrate_multi_view_apriltag_rig_result));
  }
  if (!FLAGS_output_config.empty()) {
    farm_ng::core::WriteProtobufToJsonFile(FLAGS_output_config, config);
    return 0;
  }
  LOG(ERROR) << "Please provide a config.";
  return -1;
}
int main(int argc, char* argv[]) {
  return farm_ng::core::Main(argc, argv, &Main, &Cleanup);
}
