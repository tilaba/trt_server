import numpy

class InferenceStats(object):
  """Statistics useful for evaluating basic classification and
     runtime performance"""

  @staticmethod
  def print_summary(stats, percentiles=[50, 90, 99]):
    filtered = numpy.ma.masked_invalid(stats.timings).compressed() # remove NaNs

    print '\nInference error rate: %s%%' % (
        stats.classification_error * 100)

    print "Request error rate: %s%%" % (
        (1.0 - float(filtered.size) / stats.timings.size) * 100)

    print "Avg. Throughput: %s reqs/s" % (
        float(stats.num_tests) / stats.total_elapsed_time)

    if filtered.size > 0:
      print "Request Latency (percentiles):"
      for pc, x in zip(percentiles, numpy.percentile(filtered, percentiles)):
        print "  %ith ....... %ims" % (pc, x * 1000.0)

  def __init__(self, num_tests, classification_error,
               timings, total_elapsed_time):
    assert num_tests == timings.size
    self.num_tests = num_tests
    self.classification_error = classification_error
    self.timings = timings
    self.total_elapsed_time = total_elapsed_time
