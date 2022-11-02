# Copyright 2016 IBM Corp. All Rights Reserved.
#!/usr/bin/python2.7

"""Functions for downloading and extracting pretrained py-faster-rcnn/ssd caffe models
   and client utils."""
from __future__ import print_function

import argparse
import tarfile
import os

from six.moves import urllib
from os.path import join

# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("dl_path",
  help="location to download the repository")
parser.add_argument("--type",
  dest="type", choices=['ssd', 'rcnn'], help="detector type")
parser.add_argument("--version",
  type=int, default=1, help="model version")
parser.add_argument("--export-path",
  dest="export_path", default=None, help="path to export the model to")

# -----------------------------------------------------------------------------

VERSION_FORMAT_SPECIFIER = "%08d"
CLASSES = (
  '__background__',
  'aeroplane', 'bicycle', 'bird', 'boat',
  'bottle', 'bus', 'car', 'cat', 'chair',
  'cow', 'diningtable', 'dog', 'horse',
  'motorbike', 'person', 'pottedplant',
  'sheep', 'sofa', 'train', 'tvmonitor'
)

# -----------------------------------------------------------------------------

def maybe_download(url, filename, work_directory):
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)

  filepath = join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  return filepath

def check(path):
  if not os.path.exists(path):
    raise IOError('{:s} not found.\n'.format(path))

def link(src, dst):
  print('%s => %s' % (dst, src))
  check(src)

  if not os.path.exists(dst):
    os.symlink(src, dst)

def untar(src, dst, force=False):
  if not os.path.exists(dst) or force:
    print('Extracting "%s" => "%s"' % (src, dst))
    with tarfile.open(src) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=dst)

def write_class_labels(export_dir):
  print ('Writing class labels to %s' % export_dir)
  with open(join(export_dir, 'classlabels.txt'), 'w') as f:
    for _, cls in enumerate(CLASSES):
      f.write(cls + '\n')

def create_export_dir(export_dir):
  if os.path.exists(export_dir):
    raise RuntimeError('Overwriting exports can cause corruption and are '
                       'not allowed. Duplicate export dir: %s' % export_dir)

  print('Exporting to %s' % export_dir)
  os.makedirs(export_dir)

# -----------------------------------------------------------------------------

def fetch_rcnn(base_path):
  REPO_NAME       = 'py-faster-rcnn'
  COMMIT_SHA1     = 'd14cb16b78816cc5ab0f10283381cf2ff3c6a1af'
  SOURCE_URL      = 'https://github.com/Austriker/%s/archive/%s.tar.gz' % (REPO_NAME, COMMIT_SHA1)
  DEMO_MODEL_URL  = 'http://www.cs.berkeley.edu/~rbg/faster-rcnn-data/faster_rcnn_models.tgz'
  DEMO_MODEL_FILE = 'faster_rcnn_models.tgz'

  # 1. Download py-faster-rcnn repository (as a tar.gz)
  print('Downloading repository...', SOURCE_URL)
  filepath = maybe_download(SOURCE_URL, '%s-%s.tar.gz' % (REPO_NAME, COMMIT_SHA1), base_path)
  untar(filepath, base_path, True)

  # 2. standardize the library directory structure
  src = join(base_path, '%s-%s/' % (REPO_NAME, COMMIT_SHA1))
  repo_path = join(base_path, 'rcnn')
  link(src, repo_path)

  # 3. Download demo models
  print('Downloading Faster R-CNN demo models (695M)...')
  filename = maybe_download(DEMO_MODEL_URL, DEMO_MODEL_FILE, base_path)
  data_path = join(repo_path, 'data')
  untar(filename, data_path, True)


def fetch_ssd(base_path):
  DEMO_MODEL_URL  = 'http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712_SSD_300x300.tar.gz'
  DEMO_MODEL_FILE = 'models_VGGNet_VOC0712_SSD_300x300.tar.gz'

  # 1. Download demo models
  print('Downloading SSD VGG-NET 300x300 demo models (101M)...')
  filename = maybe_download(DEMO_MODEL_URL, DEMO_MODEL_FILE, base_path)
  data_path = join(base_path, 'ssd')
  untar(filename, data_path)


def export_rcnn(base_path, export_base, export_version):
  # check some paths
  repo_path = join(base_path, 'rcnn')
  check(repo_path)

  data_path = join(repo_path, 'data')
  check(data_path)

  models_path = join(repo_path, 'models')
  check(models_path)

  # 1. Create export path
  export_dir = join(export_base, VERSION_FORMAT_SPECIFIER % export_version)
  create_export_dir(export_dir)
  write_class_labels(export_dir)

  # 2. setup symlinks (include the rcnn python lib/)
  src = join(data_path, 'faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel')
  dst = join(export_dir, 'weights.caffemodel')

  link(src, dst)

  src = join(models_path, 'pascal_voc/VGG16/faster_rcnn_alt_opt/faster_rcnn_test.pt')
  dst = join(export_dir, 'deploy.prototxt')

  link(src, dst)

  src = join(repo_path, 'lib')
  dst = join(export_base, 'lib')

  link(src, dst)


def export_ssd(base_path, export_path, export_version):
  models_path = join(base_path, 'ssd/models/VGGNet/VOC0712/SSD_300x300')
  check(models_path)

  export_dir = join(export_path, VERSION_FORMAT_SPECIFIER % export_version)
  create_export_dir(export_dir)
  write_class_labels(export_dir)

  src = join(models_path, 'VGG_VOC0712_SSD_300x300_iter_60000.caffemodel')
  dst = join(export_dir, 'weights.caffemodel')

  link(src, dst)

  src = join(models_path, 'deploy.prototxt')
  dst = join(export_dir, 'deploy.prototxt')

  link(src, dst)

if __name__ == '__main__':
  args = parser.parse_args()
  fetch_dir = args.dl_path
  export_dir = args.export_path

  fetch = None
  export = None

  if args.type == 'ssd':
    fetch = fetch_ssd
    export = export_ssd
  elif args.type == 'rcnn':
    fetch = fetch_rcnn
    export = export_rcnn
  else:
    raise ValueError('unknown detector type')

  # 1. fetch stuff
  if not os.path.exists(fetch_dir):
    os.makedirs(fetch_dir)
  fetch(fetch_dir)

  # 2. build a model export
  if export_dir != None:
    if not os.path.exists(export_dir):
      os.makedirs(export_dir)
    export(fetch_dir, export_dir, args.version)
