/* Copyright 2016 IBM Corp. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
============================================================================ */

// A gRPC server that locates and classifies objects within images.

#include <stddef.h>
#include <unistd.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <utility>

// gRPC
#include "grpc++/completion_queue.h"
#include "grpc++/security/server_credentials.h"
#include "grpc++/server.h"
#include "grpc++/server_builder.h"
#include "grpc++/server_context.h"
#include "grpc++/support/async_unary_call.h"
#include "grpc++/support/status.h"
#include "grpc++/support/status_code_enum.h"
#include "grpc/grpc.h"

// Tensor + utilities
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/contrib/session_bundle/signature.h"

// TFS
#include "tensorflow_serving/batching/basic_batch_scheduler.h"
#include "tensorflow_serving/batching/batch_scheduler.h"
#include "tensorflow_serving/core/manager.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/core/servable_id.h"

// service api
#include "tensorflow_serving/example/obj_detector.grpc.pb.h"
#include "tensorflow_serving/example/obj_detector.pb.h"

// caffe servable
#include "tensorflow_serving/servables/caffe/caffe_simple_servers.h"
#include "tensorflow_serving/servables/caffe/caffe_session_bundle.h"
#include "tensorflow_serving/servables/caffe/caffe_signature.h"

// utils (mostly to support faster-rcnn)
#include "tensorflow_serving/example/obj_detector_utils.h"
#include "tensorflow_serving/example/rpc_utils.h"

using grpc::InsecureServerCredentials;
using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using grpc::Status;
using grpc::StatusCode;

using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::strings::StrCat;
using tensorflow::serving::ClassificationSignature;

using Resolution = std::pair<int32_t /* h */, int32_t /* w */>;
using BundleType = tensorflow::serving::CaffeSessionBundle;

namespace {
const string USAGE = StrCat(
    "Usage: obj_detector --port=<port> --resolution=<HxW> [--help]  "
    "/path/to/export",
    "\n",
    "\n  --port:        The port to listen on for RPC connections, e.g. 9000",
    "\n  --resolution:  The resolution of image to accept, e.g. 800x600", "\n");

#if !defined(WITH_RCNN) && !defined(WITH_SSD)

static_assert(false,
              "\n\n INVALID BUILD CONFIGURATION: You must define a valid "
              "detector type! \n\n");

#elif WITH_RCNN

const char* TYPE_NAME = "Faster R-CNN";
const int32_t MAX_BATCH_SIZE = 1;
const int32_t BATCH_TIMEOUT = 0;
const Resolution DEFAULT_RES{600, 800};
const int NUM_CHANNELS = 3;

inline ::pixel_means_type means() {
  pixel_means_type m;
  m.setValues({{102.9801}, {115.9465}, {122.7717}});
  return m;
}

#elif WITH_SSD

const char* TYPE_NAME = "SSD";
const int32_t MAX_BATCH_SIZE = 24;
const int32_t BATCH_TIMEOUT = 1000 * 8 /* 8ms */;
const Resolution DEFAULT_RES{300, 300};
const int NUM_CHANNELS = 3;

inline ::pixel_means_type means() {
  pixel_means_type m;
  m.setValues({{104.0}, {117.0}, {123.0}});
  return m;
}

#endif

// Creates a gRPC Status from a TensorFlow Status.
Status ToGRPCStatus(const tensorflow::Status& status) {
  return Status(static_cast<grpc::StatusCode>(status.code()),
                status.error_message());
}

namespace DetectorStub {
using AsyncService = tensorflow::serving::DetectService::AsyncService;

constexpr rpc_util::UnaryRequestStub<AsyncService,
                                     tensorflow::serving::DetectRequest,
                                     tensorflow::serving::DetectResponse>
    Detect{&AsyncService::RequestDetect};

constexpr rpc_util::UnaryRequestStub<AsyncService,
                                     tensorflow::serving::ConfigurationRequest,
                                     tensorflow::serving::DetectConfiguration>
    GetConfiguration{&AsyncService::RequestGetConfiguration};
}

struct Task;
class DetectServiceImpl final {
 public:
  DetectServiceImpl(const string& servable_name,
                    const Resolution input_resolution,
                    std::unique_ptr<tensorflow::serving::Manager> manager);

  void Serve(::grpc::ServerCompletionQueue* cq,
             DetectorStub::AsyncService* stub);

 private:
  void GetConfiguration(
      decltype(DetectorStub::GetConfiguration)::Handle* call_data);

  void Detect(decltype(DetectorStub::Detect)::Handle* call_data);

  void DoDetectInBatch(std::unique_ptr<tensorflow::serving::Batch<Task>> batch);

  tensorflow::Status DoDetectInBatch_impl(
      const std::unique_ptr<tensorflow::serving::Batch<Task>>& batch);

  const tensorflow::serving::ServableRequest servable_req_;
  const Resolution input_resolution_;
  const size_t img_buffsize_;
  const ::pixel_means_type img_pixel_means_;

  std::unique_ptr<tensorflow::serving::Manager> manager_;
  std::unique_ptr<tensorflow::serving::BasicBatchScheduler<Task>>
      batch_scheduler_;
};

// A Task holds all of the information for a single inference request.
struct Task : public tensorflow::serving::BatchTask {
  ~Task() override = default;
  size_t size() const override { return 1; }

  Task(decltype(DetectorStub::Detect)::Handle* calldata_arg)
      : calldata(calldata_arg) {}

  decltype(DetectorStub::Detect)::Handle* calldata;
};

DetectServiceImpl::DetectServiceImpl(
    const string& servable_name, const Resolution input_resolution,
    std::unique_ptr<tensorflow::serving::Manager> manager)
    : servable_req_(
          tensorflow::serving::ServableRequest::Latest(servable_name)),
      input_resolution_(input_resolution),
      img_buffsize_(NUM_CHANNELS * std::get<0>(input_resolution) *
                    std::get<1>(input_resolution)),
      img_pixel_means_(means()),
      manager_(std::move(manager)) {
  tensorflow::serving::BasicBatchScheduler<Task>::Options sched_opts;
  sched_opts.thread_pool_name = "detector_batch_threads";
  sched_opts.max_enqueued_batches = 250 / MAX_BATCH_SIZE;
  sched_opts.max_batch_size = MAX_BATCH_SIZE;
  sched_opts.batch_timeout_micros = BATCH_TIMEOUT;
  TF_CHECK_OK(tensorflow::serving::BasicBatchScheduler<Task>::Create(
      sched_opts,
      [this](std::unique_ptr<tensorflow::serving::Batch<Task>> batch) {
        this->DoDetectInBatch(std::move(batch));
      },
      &batch_scheduler_));
}

void DetectServiceImpl::Serve(::grpc::ServerCompletionQueue* cq,
                              DetectorStub::AsyncService* s) {
  rpc_util::begin_async_unary(
      this, DetectorStub::Detect, s, cq,
      [](DetectServiceImpl* impl,
         decltype(DetectorStub::Detect)::Handle* data) { impl->Detect(data); });

  rpc_util::begin_async_unary(
      this, DetectorStub::GetConfiguration, s, cq,
      [](DetectServiceImpl* impl,
         decltype(DetectorStub::GetConfiguration)::Handle* data) {
        impl->GetConfiguration(data);
      });
}

void DetectServiceImpl::GetConfiguration(
    decltype(DetectorStub::GetConfiguration)::Handle* call_data) {
  int32_t w, h;
  std::tie(w, h) = input_resolution_;

  auto in_spec = call_data->mutable_response()->mutable_input_image_shape();
  in_spec->Add(NUM_CHANNELS);
  in_spec->Add(h);
  in_spec->Add(w);

  call_data->Finish(Status::OK);
}

void DetectServiceImpl::Detect(
    decltype(DetectorStub::Detect)::Handle* calldata) {
  // Verify input.
  if (calldata->request().image_data().size() != img_buffsize_) {
    calldata->Finish(Status(
        StatusCode::INVALID_ARGUMENT,
        tensorflow::strings::StrCat("expected image_data of size ",
                                    img_buffsize_, ", got ",
                                    calldata->request().image_data().size())));
    return;
  }
  // Create and submit a task to the batch scheduler.
  std::unique_ptr<Task> task(new Task(calldata));
  tensorflow::Status status = batch_scheduler_->Schedule(&task);

  if (!status.ok()) {
    calldata->Finish(ToGRPCStatus(status));
    return;
  }
}

void DetectServiceImpl::DoDetectInBatch(
    std::unique_ptr<tensorflow::serving::Batch<Task>> batch) {
  // either the entire batch succeeds or fails
  Status status = ToGRPCStatus(DoDetectInBatch_impl(batch));
  // complete the task with the given status
  for (int i = 0; i < batch->num_tasks(); i++) {
    Task* task = batch->mutable_task(i);
    task->calldata->Finish(status);
  }
}

tensorflow::Status DetectServiceImpl::DoDetectInBatch_impl(
    const std::unique_ptr<tensorflow::serving::Batch<Task>>& batch) {
  batch->WaitUntilClosed();
  if (batch->empty()) {
    return tensorflow::Status::OK();
  }

  const int batch_size = batch->num_tasks();
  std::vector<float> batch_min_thresholds(batch_size);

  Tensor im_blob(tensorflow::DT_FLOAT, {batch_size, (long)img_buffsize_});
  {
    auto dst = im_blob.flat_outer_dims<float>().data();
    for (int i = 0; i < batch_size; ++i) {
      const auto& req = batch->mutable_task(i)->calldata->request();
      std::transform(req.image_data().begin(), req.image_data().end(), dst,
                     [](const uint8_t& a) { return static_cast<float>(a); });
      dst += img_buffsize_;
      batch_min_thresholds[i] = req.min_score_threshold();
    }
  }

  int32_t w, h;
  std::tie(w, h) = input_resolution_;

  tensorflow::serving::ServableHandle<BundleType> bundle;
  TF_RETURN_IF_ERROR(manager_->GetServableHandle(servable_req_, &bundle));
  TF_RETURN_IF_ERROR(::BatchMeansSubtract(img_pixel_means_, &im_blob));

  std::vector<std::vector<::ObjDetection>> dets(batch_size);
  tensorflow::Tensor class_labels;
  tensorflow::Tensor scores;

#if WITH_RCNN
  {
    assert(batch_size == 1);
    tensorflow::Tensor boxes;
    Tensor im_info(tensorflow::DT_FLOAT, {1, 3});
    {
      auto dst = im_info.flat<float>().data();
      dst[0] = h;
      dst[1] = w;
      dst[2] = 1.0 /* scale */;
    }

    TF_RETURN_IF_ERROR(rcnn::RunClassification(im_blob, im_info,
                                               bundle->session.get(), &boxes,
                                               &scores, &class_labels));

    TF_RETURN_IF_ERROR(rcnn::ProcessDetections(
        &boxes, &scores, batch_min_thresholds[0], &dets[0]));
  }
#elif WITH_SSD
  {
    tensorflow::serving::ClassificationSignature signature;
    TF_RETURN_IF_ERROR(
        GetClassificationSignature(bundle->meta_graph_def, &signature));
    {
      std::vector<Tensor> outputs;
      TF_RETURN_IF_ERROR(bundle->session.get()->Run(
          {{signature.input().tensor_name(), im_blob}},
          {signature.classes().tensor_name(), signature.scores().tensor_name()},
          {}, &outputs));

      class_labels = outputs[0];
      scores = outputs[1];
    }

    const int n = scores.dim_size(2);
    const auto out_mat = scores.shaped<float, 2>({n, 7});

    for (int i = 0; i < n; ++i) {
      int image_id = static_cast<int>(out_mat(i, 0));
      ::ObjDetection det{
          std::array<int, 4>{static_cast<int>(out_mat(i, 3) * w),
                             static_cast<int>(out_mat(i, 4) * h),
                             static_cast<int>(out_mat(i, 5) * w),
                             static_cast<int>(out_mat(i, 6) * h)},
          static_cast<int>(out_mat(i, 1)), out_mat(i, 2)};
      dets[image_id].push_back(det);
    }
  }
#endif

  const auto labels_mat = class_labels.matrix<string>();
  for (int i = 0; i < batch_size; ++i) {
    auto calldata = batch->mutable_task(i)->calldata;

    decltype(DetectorStub::Detect)::Response* resp =
        calldata->mutable_response();

    for (const auto& det : dets[i]) {
      if (det.class_idx == 0) continue;

      if (det.score < batch_min_thresholds[i]) continue;

      tensorflow::serving::Detection* det_proto = resp->add_detections();
      det_proto->set_roi_x1(det.roi_rect[0]);
      det_proto->set_roi_y1(det.roi_rect[1]);
      det_proto->set_roi_x2(det.roi_rect[2]);
      det_proto->set_roi_y2(det.roi_rect[3]);
      det_proto->set_score(det.score);
      det_proto->set_class_label(labels_mat(0, det.class_idx));
    }
  }
  return tensorflow::Status::OK();
}

void HandleRpcs(DetectServiceImpl* service_impl,
                DetectorStub::AsyncService* service,
                ServerCompletionQueue* cq) {
  service_impl->Serve(cq, service);

  void* tag;  // uniquely identifies a request.
  bool ok;
  while (true) {
    // Block waiting to read the next event from the completion queue. The
    // event is uniquely identified by its tag, which in this case is the
    // memory address of a CallData instance.
    cq->Next(&tag, &ok);
    GPR_ASSERT(ok);
    static_cast<rpc_util::RpcHandleBase*>(tag)->Proceed();
  }
}

// Runs DetectService server until shutdown.
void RunServer(const int port, const string& servable_name,
               const Resolution resolution,
               std::unique_ptr<tensorflow::serving::Manager> manager) {
  // "0.0.0.0" is the way to listen on localhost in gRPC.
  const string server_address = "0.0.0.0:" + std::to_string(port);

  DetectorStub::AsyncService service;
  ServerBuilder builder;
  std::shared_ptr<grpc::ServerCredentials> creds = InsecureServerCredentials();
  builder.AddListeningPort(server_address, creds);
  builder.RegisterService(&service);
  std::unique_ptr<ServerCompletionQueue> cq = builder.AddCompletionQueue();
  std::unique_ptr<Server> server(builder.BuildAndStart());
  LOG(INFO) << "Running...";

  DetectServiceImpl service_impl(servable_name, resolution, std::move(manager));
  HandleRpcs(&service_impl, &service, cq.get());
}

string proc_path() {
  char result[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  return string(result, (count > 0) ? count : 0);
}

void parse_cmdline(int& argc, char** argv, tensorflow::int32& port,
                   Resolution& resolution, std::string& export_base_path) {
  std::string resolution_str;
  const bool parse_result = tensorflow::Flags::Parse(
      &argc, argv, {tensorflow::Flag("port", &port, "port to listen on"),
                    tensorflow::Flag("resolution", &resolution_str, "input image resolution")});

  if (!parse_result) {
    LOG(FATAL) << "Error parsing command line flags.";
  }
  if (resolution_str.size() > 0) {
    std::stringstream ss(resolution_str);
    std::vector<int32_t> dims;

    while (ss.good()) {
      std::string substr;
      std::getline(ss, substr, 'x');
      int32_t dim = std::atoi(substr.c_str());
      if (dim <= 0) {
        LOG(FATAL) << "Error parsing resolution dimension: " << dim;
      }
      dims.push_back(dim);
    }
    if (dims.size() != 2) {
      LOG(FATAL) << "Invalid image resolution";
    }
    resolution = std::make_pair(dims[0] /* h */, dims[1] /* w */);
  } else {
    resolution = DEFAULT_RES;
  }
  if (argc != 2) {
    LOG(FATAL) << USAGE;
  }

  export_base_path = argv[1];
}
}  // namespace

int main(int argc, char** argv) {
  int32_t port;
  Resolution res;
  std::string export_base_path;
  parse_cmdline(argc, argv, port, res, export_base_path);

  LOG(INFO) << "Detector type: " << TYPE_NAME
            << "\n  Input resolution: h=" << std::get<0>(res)
            << "px, w=" << std::get<1>(res) << "px";

  // Initialize Caffe subsystem
  tensorflow::serving::CaffeGlobalInit(&argc, &argv);
  tensorflow::serving::CaffeSourceAdapterConfig source_adapter_config;
  {
    auto bundle_cfg = source_adapter_config.mutable_config();
    tensorflow::TensorShapeProto* shape;

#if WITH_RCNN
    {
      // enable pycaffe
      bundle_cfg->set_enable_py_caffe(true);
      // add path to pycaffe python module(s)
      bundle_cfg->add_python_path(StrCat(
          proc_path(),
          ".runfiles/tf_serving/tensorflow_serving/servables/caffe/pycaffe"));
      // add path to py-faster-rcnn
      bundle_cfg->add_python_path(export_base_path + "/lib");
      shape = &(*bundle_cfg->mutable_named_initial_shapes())["data"];
    }
#elif WITH_SSD
    shape = bundle_cfg->mutable_initial_shape();
#endif
    // reshape the network for the given input resolution.
    assert(shape != nullptr);
    shape->add_dim()->set_size(1);
    shape->add_dim()->set_size(NUM_CHANNELS);
    shape->add_dim()->set_size(std::get<0>(res));
    shape->add_dim()->set_size(std::get<1>(res));
  }
  std::unique_ptr<tensorflow::serving::Manager> manager;
  tensorflow::Status status = tensorflow::serving::simple_servers::
      CreateSingleCaffeModelManagerFromBasePath(
          export_base_path, source_adapter_config, &manager);

  TF_CHECK_OK(status) << "Error creating manager";

  // Wait until at least one model is loaded.
  std::vector<tensorflow::serving::ServableId> ready_ids;
  // TODO(b/25545573): Create a more streamlined startup mechanism than polling.
  do {
    LOG(INFO) << "Waiting for models to be loaded...";
    tensorflow::Env::Default()->SleepForMicroseconds(1 * 1000 * 1000 /*1 sec*/);
    ready_ids = manager->ListAvailableServableIds();
  } while (ready_ids.empty());

  // Run the service.
  RunServer(port, ready_ids[0].name, res, std::move(manager));
  return 0;
}
