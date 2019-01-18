def if_pycaffe(if_true, if_false = []):
  return select({
      "@caffe_tools//:caffe_python_layer": if_true,
      "//conditions:default": if_false
  })

def caffe_pkg(label):
  return select({
      "//conditions:default": ["@caffe//" + label],
      "@caffe_tools//:use_caffe_rcnn": ["@caffe_rcnn//" + label],
      "@caffe_tools//:use_caffe_ssd": ["@caffe_ssd//" + label],
  })
