# TensorFlow Serving external dependencies that can be loaded in WORKSPACE
# files.

load('@org_tensorflow//tensorflow:workspace.bzl', 'tf_workspace')

# All TensorFlow Serving external dependencies.
# workspace_dir is the absolute path to the TensorFlow Serving repo. If linked
# as a submodule, it'll likely be '__workspace_dir__ + "/serving"'
def tf_serving_workspace(workspace_dir):
  native.local_repository(
    name = "inception_model",
    path = "tf_models/inception",
  )

  tf_workspace(path_prefix = "", tf_repo_name = "org_tensorflow")

  # ===== gRPC dependencies =====
  native.bind(
    name = "libssl",
    actual = "@boringssl//:ssl",
  )

  native.bind(
    name = "zlib",
    actual = "@zlib_archive//:zlib",
  )

  # ===== caffe flavors =====
  native.new_git_repository(
    name = "caffe",
    remote = "https://github.com/BVLC/caffe",
    commit = "50c9a0fc8eed0101657e9f8da164d88f66242aeb",
    init_submodules = True,
    build_file = workspace_dir + "/caffe.BUILD",
  )

  native.new_git_repository(
    name = "caffe_rcnn",
    remote = "https://github.com/Austriker/caffe",
    commit = "3c27a70ae7304feb48c60b268c70adf585879d50",
    init_submodules = True,
    build_file = workspace_dir + "/caffe.BUILD",
  )

  native.new_git_repository(
    name = "caffe_ssd",
    remote = "https://github.com/weiliu89/caffe",
    commit = "781288500b34a532fd37ee0288710a0f012d2b6c",
    init_submodules = True,
    build_file = workspace_dir + "/caffe.BUILD",
  )

  # ===== caffe build/integration tools =====
  native.local_repository(
    name = "caffe_tools",
    path = workspace_dir + "/third_party/caffe"
  )
