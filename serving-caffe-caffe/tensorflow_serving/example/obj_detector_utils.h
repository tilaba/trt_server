/* Copyright IBM Corp. All Rights Reserved. */

#ifndef TENSORFLOW_SERVING_EXAMPLE_OBJ_DETECTOR_UTILS_H_
#define TENSORFLOW_SERVING_EXAMPLE_OBJ_DETECTOR_UTILS_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/public/session.h"

#include <vector>
#include <array>

// 3-channel pixel mean representation
using pixel_means_type =
    Eigen::TensorFixedSize<float, Eigen::Sizes<3, 1>, Eigen::RowMajor>;

// apply pixel mean subtraction to a batch of
// images in planar format.
tensorflow::Status BatchMeansSubtract(const pixel_means_type& means,
                                      tensorflow::Tensor* im_batch_blob);

// canonical representation of an object detection
struct ObjDetection {
  std::array<int, 4> roi_rect;
  int class_idx;
  float score;

  inline std::string DebugString() const {
    return tensorflow::strings::StrCat("ObjDetection { class: ", class_idx,
                                       ", score: ", score, ", rect: [",
                                       roi_rect[0], " ", roi_rect[1], " ",
                                       roi_rect[2], " ", roi_rect[3], "] }");
  }
};

// faster-rcnn specific utils
namespace rcnn {
tensorflow::Status RunClassification(const tensorflow::Tensor& im_blob,
                                     const tensorflow::Tensor& im_info,
                                     tensorflow::Session* session,
                                     tensorflow::Tensor* pred_boxes,
                                     tensorflow::Tensor* scores,
                                     tensorflow::Tensor* class_labels);

tensorflow::Status ProcessDetections(const tensorflow::Tensor* pred_boxes,
                                     const tensorflow::Tensor* scores,
                                     const float detection_threshold,
                                     std::vector<ObjDetection>* dets);

}  // namespace rcnn

#endif // TENSORFLOW_SERVING_EXAMPLE_OBJ_DETECTOR_UTILS_H_