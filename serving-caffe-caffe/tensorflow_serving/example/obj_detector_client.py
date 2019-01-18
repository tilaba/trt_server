# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""A client that talks to rcnn_detector service. Please see
rcnn_detector.proto for service API details.

Typical usage example:
    rcnn_client.py --server=localhost:9000 --gui --img=/path/to/image.jpg
"""

import sys
import threading
from os import listdir
from os.path import isfile, join
from timeit import default_timer as timer

import numpy
import tensorflow as tf
import skimage.io as io
import skimage.transform as transform
import matplotlib.pyplot as plt

from grpc.beta import implementations
from grpc.framework.interfaces.face.face import AbortionError
from tensorflow_serving.example import obj_detector_pb2
from client_util import InferenceStats

tf.app.flags.DEFINE_integer(
  'concurrency', 1,'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer(
  'num_tests', -1, 'Number of test images')
tf.app.flags.DEFINE_string(
  'server', '', 'obj_detector service host:port')
tf.app.flags.DEFINE_bool(
  'verbose', False, 'print detections to stdout')
tf.app.flags.DEFINE_bool(
  'gui', False, 'show detections in a gui')
tf.app.flags.DEFINE_string(
  'img', 'https://upload.wikimedia.org/wikipedia/commons/4/4f/G8_Summit_working_session_on_global_and_economic_issues_May_19%2C_2012.jpg',
  'url or path of an image to classify')
tf.app.flags.DEFINE_string(
  'imgdir', None, 'path to a gallery of images')

FLAGS = tf.app.flags.FLAGS

def connect(hostport):
  """
  Connect to the inference server
  """
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  return obj_detector_pb2.beta_create_DetectService_stub(channel)

def handshake(stub):
  """
  retrieve service-specific cofigurations such as 
  input-image shape, number of channels etc.
  """
  req = obj_detector_pb2.ConfigurationRequest()
  result = stub.GetConfiguration(req, 5)
  assert len(result.input_image_shape) == 3 # this client only supports rgb
  
  # return the input shape
  return (result.input_image_shape[0],
          result.input_image_shape[1],
          result.input_image_shape[2])

def im_transpose(im):
  """
  transpose input image to shape [c = 3, h, w] and BGR channel order.
  """
  R, G, B = im.transpose((2, 0, 1))
  return numpy.array((B,G,R), dtype=numpy.uint8)

def im_scale_to_fit(im, out_shape):
  """
  rescale an input image of arbitary dimensions to fit
  within out_shape, maintaining aspect ratio and not
  permitting cropping.

  Returns the rescaled image
  """
  h, w, c = im.shape
  out_c, out_h, out_w = out_shape

  assert(out_c == 3)

  scale = min(float(out_h) / h, float(out_w) / w)
  im2 = transform.resize(im, numpy.floor((h * scale, w * scale)),
                         preserve_range=True)

  result = numpy.zeros([out_h, out_w, c], dtype=numpy.uint8)
  result[0:im2.shape[0], 0:im2.shape[1], ...] = im2
  return result

def do_inference(stub, concurrency, num_tests, images, detection_thresh=0.7):
  cv = threading.Condition()
  
  result = {'active': 0, 'error': 0, 'done': 0}
  result_dets = []
  result_timing = numpy.zeros(num_tests, dtype=numpy.float64);

  def done(reqid, im_idx, result_future):
    with cv:
      try:
        res = result_future.result()
        result_timing[reqid] = timer() - result_timing[reqid]
        result_dets.append((im_idx, res.detections))
        
        sys.stdout.write('.')
        sys.stdout.flush()

      except AbortionError as e:
        result_timing[reqid] = numpy.NaN
        result['error'] += 1
        print ("An RPC error occured: %s" % e)

      result['done'] += 1
      result['active'] -= 1
      cv.notify()
  
  start_time = timer()
  for n in range(num_tests):
    im_idx = n % len(images)
    im = images[im_idx]
    im_input = im_transpose(im)

    req = obj_detector_pb2.DetectRequest(
        image_data=bytes(im_input.data),
        min_score_threshold=detection_thresh)

    with cv:
      while result['active'] == concurrency:
        cv.wait()
      result['active'] += 1

    result_timing[n] = timer()
    result_future = stub.Detect.future(req, 30)
    result_future.add_done_callback(
        lambda result_future, n=n, im_idx=im_idx: done(n, im_idx, result_future))

  with cv:
    while result['done'] != num_tests:
      cv.wait()

  stats = InferenceStats(num_tests,
      result['error'] / float(num_tests),
      result_timing,
      timer() - start_time)

  return (stats, result_dets)

def vis_detections(im, dets):
  """
  Draw image and detected bounding boxes.
  """
  fig, ax = plt.subplots(figsize=(12, 12))
  ax.imshow(im, aspect='equal')
  
  for det in dets:
    ax.text(
        det.roi_x1+3, det.roi_y1-3, "{:s} ({:.3f})".format(det.class_label, det.score),
        color='black', backgroundcolor='white',
        verticalalignment='bottom')

    ax.add_patch(
        plt.Rectangle(
            (det.roi_x1, det.roi_y1),
            det.roi_x2 - det.roi_x1,
            det.roi_y2 - det.roi_y1,
            fill=False, edgecolor='white',
            linewidth=2))
  plt.tight_layout()
  plt.draw()
  plt.show()

def main(_):
  if not FLAGS.server:
    print 'please specify server host:port'
    return

  # build image list
  paths = None
  if FLAGS.imgdir != None:
    paths = [join(FLAGS.imgdir, f) for f in listdir(FLAGS.imgdir) if isfile(join(FLAGS.imgdir, f))]
  else:
    paths = [FLAGS.img]

  # connect and get input image shape
  stub = connect(FLAGS.server)
  input_shape = handshake(stub)

  print ("Connected. Input Shape: {0}".format(input_shape))

  # load the image gallery 
  ims = [im_scale_to_fit(io.imread(path), input_shape) for path in paths]
  n = len(ims) if FLAGS.num_tests < 0 else FLAGS.num_tests

  # run the tests
  stats, results = do_inference(stub, FLAGS.concurrency, n, ims)
  InferenceStats.print_summary(stats)

  # verbose output / gui
  for image_idx, dets in results:
    if FLAGS.verbose:
      print('\n{0}\n----------------'.format(paths[image_idx]))
      for det in dets:
        print(' {:s}\n  > score: {:.3f}\n  > bbox (x1, y1, x2, y2): ({:d}, {:d}, {:d}, {:d})\n'.format(
              det.class_label, det.score, det.roi_x1, det.roi_y1, det.roi_x2, det.roi_y2))
    if FLAGS.gui:
      vis_detections(ims[image_idx], dets)

if __name__ == '__main__':
  tf.app.run()
