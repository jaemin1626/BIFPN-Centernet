from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from .exdet import ExdetDetector
# from .ddd import DddDetector
# from .multi_pose import MultiPoseDetector
from .ctdet import CtdetDetector

detector_factory = {
    # 'exdet': ExdetDetector,
    # 'ddd': DddDetector,
    'ctdet': CtdetDetector,
    # 'multi_pose': MultiPoseDetector,
}
