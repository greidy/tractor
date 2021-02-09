#ifndef FARM_NG_PERCEPTION_TENSOR_H_
#define FARM_NG_PERCEPTION_TENSOR_H_
#include <Eigen/Core>
#include <string>

#include "farm_ng/perception/tensor.pb.h"

namespace farm_ng::perception {

inline void EigenToTensor(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::ColMajor>& x,
    std::string rows_name, std::string cols_name, Tensor* out) {
  out->set_dtype(Tensor::DATA_TYPE_FLOAT64);
  auto dim1 = out->add_shape();
  dim1->set_size(x.rows());
  dim1->set_name(rows_name);

  auto dim2 = out->add_shape();
  dim2->set_size(x.cols());
  dim2->set_name(cols_name);

  out->set_data(reinterpret_cast<const char*>(x.data()),
                x.size() * sizeof(double));
}

inline Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
TensorToEigenMapXd(const farm_ng::perception::Tensor& x) {
  CHECK_EQ(x.dtype(), Tensor::DATA_TYPE_FLOAT64);
  CHECK_EQ(x.shape_size(), 2);
  int rows = x.shape(0).size();
  int cols = x.shape(1).size();
  return Eigen::Map<
      const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>(
      reinterpret_cast<const double*>(x.data().data()), rows, cols);
}

inline void TensorToEigen(const farm_ng::perception::Tensor& x,
                          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::ColMajor>* out) {
  *out = TensorToEigenMapXd(x);
}
}  // namespace farm_ng::perception

#endif
