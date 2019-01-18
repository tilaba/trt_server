#include "tensorflow_serving/example/obj_detector_utils.h"

#include <stddef.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <functional>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"

using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::TensorShape;

// for each image in the minibatch, perform means subtraction
tensorflow::Status BatchMeansSubtract(const pixel_means_type& bgr_mean,
                                      Tensor* im_batch_blob) {
  using namespace Eigen;

  auto im_batch_bgr = im_batch_blob->matrix<float>();
  if (im_batch_bgr.dimension(1) % 3 != 0) {
    return tensorflow::errors::Internal(
        tensorflow::strings::StrCat("given batch isn't bgr image data"));
  }

  int batch_size = im_batch_bgr.dimension(0);
  int im_buff_size = im_batch_bgr.dimension(1);

  for (auto i = 0; i < batch_size; ++i) {
    DSizes<ptrdiff_t, 2> off(i, 0);
    DSizes<ptrdiff_t, 2> ext(1, im_buff_size);

    // reshape buffer to 2d-planar
    Eigen::Tensor<float, 2>::Dimensions dim2(3, im_buff_size / 3);
    auto im_bgr = im_batch_bgr.slice(off, ext).reshape(dim2);

    // apply mean-subtraction
    im_bgr -= bgr_mean.broadcast(Eigen::array<int64_t, 2>{{1, dim2[1]}});
  }
  return tensorflow::Status::OK();
}

namespace {
const std::vector<string>* output_tensor_names() {
  static std::vector<string> output{"cls_prob", "bbox_pred", "rois",
                                    "__labels__"};
  return &output;
}

static inline void DecreasingArgSort(const std::vector<float>& values,
                                     std::vector<int>* indices) {
  indices->resize(values.size());
  for (int i = 0; i < values.size(); ++i) (*indices)[i] = i;
  std::sort(
      indices->begin(), indices->end(),
      [&values](const int i, const int j) { return values[i] > values[j]; });
}

// Compute intersection-over-union overlap between boxes i and j.
static inline float ComputeIOU(
    const Eigen::Tensor<float, 2, Eigen::RowMajor>& boxes, int i, int j) {
  const float xmin_i = std::min<float>(boxes(i, 0), boxes(i, 2));
  const float ymin_i = std::min<float>(boxes(i, 1), boxes(i, 3));
  const float xmax_i = std::max<float>(boxes(i, 0), boxes(i, 2));
  const float ymax_i = std::max<float>(boxes(i, 1), boxes(i, 3));
  const float xmin_j = std::min<float>(boxes(j, 0), boxes(j, 2));
  const float ymin_j = std::min<float>(boxes(j, 1), boxes(j, 3));
  const float xmax_j = std::max<float>(boxes(j, 0), boxes(j, 2));
  const float ymax_j = std::max<float>(boxes(j, 1), boxes(j, 3));

  const float area_i = (ymax_i - ymin_i) * (xmax_i - xmin_i);
  const float area_j = (ymax_j - ymin_j) * (xmax_j - xmin_j);

  if (area_i <= 0 || area_j <= 0) return 0.0;

  const float intersection_ymin = std::max<float>(ymin_i, ymin_j);
  const float intersection_xmin = std::max<float>(xmin_i, xmin_j);
  const float intersection_ymax = std::min<float>(ymax_i, ymax_j);
  const float intersection_xmax = std::min<float>(xmax_i, xmax_j);
  const float intersection_area =
      std::max<float>(intersection_ymax - intersection_ymin, 0.0) *
      std::max<float>(intersection_xmax - intersection_xmin, 0.0);

  return intersection_area / (area_i + area_j - intersection_area);
}

static inline void nms(
    const Eigen::Tensor<float, 2, Eigen::RowMajor>& boxes_data,  // [N, 4]
    const Eigen::Tensor<float, 2, Eigen::RowMajor>& scores,      // [N, 1]
    const float iou_threshold, const float min_score_threshold,
    const int max_output_size, std::vector<int>& selected_indices) {
  selected_indices.clear();

  int num_boxes = boxes_data.dimension(0);
  const int output_size = std::min(max_output_size, num_boxes);

  std::vector<float> scores_data(num_boxes);
  std::copy_n(scores.data(), num_boxes, scores_data.begin());

  std::vector<int> sorted_indices;
  DecreasingArgSort(scores_data, &sorted_indices);

  for (int i = 0; i < num_boxes; i++) {
    if (scores_data[sorted_indices[i]] < min_score_threshold) {
      num_boxes = i;
      break;
    }
  }

  std::vector<bool> active(num_boxes, true);
  int num_active = active.size();

  for (int i = 0; i < num_boxes; ++i) {
    if (num_active == 0 || selected_indices.size() >= (size_t)output_size)
      break;
    if (active[i]) {
      selected_indices.push_back(sorted_indices[i]);
    } else {
      continue;
    }
    for (int j = i + 1; j < num_boxes; ++j) {
      if (active[j]) {
        float iou =
            ComputeIOU(boxes_data, sorted_indices[i], sorted_indices[j]);
        if (iou > iou_threshold) {
          active[j] = false;
          num_active--;
        }
      }
    }
  }
}
}  // namespace

namespace rcnn {

tensorflow::Status clip_boxes(Tensor* boxes, const TensorShape im_shape) {
  // sanity
  if (boxes->dims() != 2 || im_shape.dims() != 2) {
    tensorflow::errors::Internal("incorrect input shapes");
  }
  if (boxes->dim_size(1) % 4 != 0) {
    tensorflow::errors::Internal(
        "incorrect shape, expected: [N,M] where M % 4 == 0");
  }

  float max_h = (float)im_shape.dim_size(0);
  float max_w = (float)im_shape.dim_size(1);

  auto box_mat = boxes->matrix<float>();
  int rows = boxes->dim_size(0);
  int classes = boxes->dim_size(1) / 4;

  for (int r = 0; r < rows; ++r) {
    for (int cls = 0; cls < classes; ++cls) {
      int off = cls * 4;

      float& x1 = box_mat(r, off + 0);
      float& y1 = box_mat(r, off + 1);
      float& x2 = box_mat(r, off + 2);
      float& y2 = box_mat(r, off + 3);

      if (x1 < 0) x1 = 0;
      if (y1 < 0) y1 = 0;
      if (x2 > max_w) x2 = max_w;
      if (y2 > max_h) y2 = max_h;
    }
  }
  return tensorflow::Status::OK();
}

// rois blob: holds R regions of interest, each is a 5-tuple
// (n, x1, y1, x2, y2) specifying an image batch index n and a
// rectangle (x1, y1, x2, y2)
static inline tensorflow::Status bbox_transform_inv(const Tensor& rois_blob,
                                                    const Tensor& deltas,
                                                    Tensor* pred_boxes) {
  // sanity
  if (rois_blob.dims() != 2 || deltas.dims() != 2) {
    tensorflow::errors::Internal("incorrect input shapes");
  }
  if (rois_blob.dim_size(1) != 5) {
    tensorflow::errors::Internal("bbox's must have shape [K,5]");
  }
  if (deltas.dim_size(1) % 4 != 0) {
    tensorflow::errors::Internal(
        "incorrect shape, expected: [N,M] where M % 4 == 0");
  }
  // zero rois_blob
  if (rois_blob.dim_size(0) == 0) {
    Tensor out(deltas.dtype(), TensorShape({deltas.dim_size(1)}));
    CHECK_EQ(pred_boxes->CopyFrom(out, out.shape()), true);
    return tensorflow::Status::OK();
  }

  Tensor out(deltas.dtype(), deltas.shape());

  auto box_mat = rois_blob.matrix<float>();
  auto deltas_mat = deltas.matrix<float>();
  auto out_mat = out.matrix<float>();

  for (int r = 0; r < rois_blob.dim_size(0); ++r) {
    float w = box_mat(r, 3) - box_mat(r, 1) + 1.0;
    float h = box_mat(r, 4) - box_mat(r, 2) + 1.0;
    float ctr_x = box_mat(r, 1) + 0.5 * w;
    float ctr_y = box_mat(r, 2) + 0.5 * h;

    for (int cls = 0; cls < deltas.dim_size(1) / 4; ++cls) {
      int off = cls * 4;

      float dx = deltas_mat(r, off + 0);
      float dy = deltas_mat(r, off + 1);
      float dw = deltas_mat(r, off + 2);
      float dh = deltas_mat(r, off + 3);

      float pred_ctr_x = dx * w + ctr_x;
      float pred_ctr_y = dy * h + ctr_y;
      float pred_w = std::exp(dw) * w;
      float pred_h = std::exp(dh) * h;

      out_mat(r, off + 0) = pred_ctr_x - 0.5 * pred_w;  // x1
      out_mat(r, off + 1) = pred_ctr_y - 0.5 * pred_h;  // y1
      out_mat(r, off + 2) = pred_ctr_x + 0.5 * pred_w;  // x2
      out_mat(r, off + 3) = pred_ctr_y + 0.5 * pred_h;  // y2
    }
  }

  CHECK_EQ(pred_boxes->CopyFrom(out, out.shape()), true);
  return tensorflow::Status::OK();
}

tensorflow::Status RunClassification(const Tensor& im_blob,
                                     const Tensor& im_info,
                                     tensorflow::Session* session,
                                     Tensor* pred_boxes, Tensor* scores,
                                     Tensor* class_labels) {
  // Run the graph with our inputs and outputs.
  std::vector<Tensor> outputs;
  const std::vector<string>* output_names = output_tensor_names();
  const tensorflow::Status run_status = session->Run(
      {
          {"data", im_blob}, {"im_info", im_info},
      },
      *output_names, {}, &outputs);

  if (!run_status.ok()) {
    return run_status;
  }
  // check the output shape
  if (outputs.size() != output_names->size()) {
    return tensorflow::errors::Internal(tensorflow::strings::StrCat(
        "Expected ", output_names->size(), " output tensor(s).  Got: ",
        outputs.size()));
  }
  // cls_prob
  CHECK_EQ(scores->CopyFrom(outputs[0], outputs[0].shape()), true);
  // class labels
  CHECK_EQ(class_labels->CopyFrom(outputs[3], outputs[3].shape()), true);

  Tensor& box_deltas = outputs[1];
  Tensor& rois = outputs[2];
  // apply bounding box regression deltas
  TF_RETURN_IF_ERROR(bbox_transform_inv(rois, box_deltas, pred_boxes));
  // clip bboxes to image dimensions
  TF_RETURN_IF_ERROR(clip_boxes(
      pred_boxes, {im_info.flat<float>()(0), im_info.flat<float>()(1)}));
  // complete
  return tensorflow::Status::OK();
}

tensorflow::Status ProcessDetections(const Tensor* pred_boxes,
                                     const Tensor* scores,
                                     const float detection_threshold,
                                     std::vector<ObjDetection>* dets) {
  using namespace Eigen;

  auto pred_boxes_mat = pred_boxes->matrix<float>();
  auto scores_mat = scores->matrix<float>();

  int n_preds = scores_mat.dimension(0);
  int n_classes = scores_mat.dimension(1);

  Eigen::Tensor<float, 2, RowMajor> tmp_boxes;
  Eigen::Tensor<float, 2, RowMajor> tmp_scores;

  std::vector<int> sorted_selected_indices;
  for (int cls = 0; cls < n_classes; cls++) {
    { /* [N, 4] */
      DSizes<ptrdiff_t, 2> off(0, cls * 4);
      DSizes<ptrdiff_t, 2> ext(n_preds, 4);

      tmp_boxes = pred_boxes_mat.slice(off, ext);
    }
    { /* [N, 1] */
      DSizes<ptrdiff_t, 2> off(0, cls);
      DSizes<ptrdiff_t, 2> ext(n_preds, 1);

      tmp_scores = scores_mat.slice(off, ext);
    }

    nms(tmp_boxes, tmp_scores, 0.3, detection_threshold, 128,
        sorted_selected_indices);

    for (auto idx : sorted_selected_indices) {
      dets->emplace_back(
          ObjDetection{std::array<int, 4>{static_cast<int>(tmp_boxes(idx, 0)),
                                          static_cast<int>(tmp_boxes(idx, 1)),
                                          static_cast<int>(tmp_boxes(idx, 2)),
                                          static_cast<int>(tmp_boxes(idx, 3))},
                       cls, tmp_scores(idx, 0)});
    }
  }
  return tensorflow::Status::OK();
}
}  // namespace rcnn
