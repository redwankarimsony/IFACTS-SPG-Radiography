# Import all the modules in utils/ folder
from .gpu_utils import get_gpu_with_least_memory_over_period
from .fusion_utils import fuse_features_with_norm 
from .verification_utils import get_verification_pairs, getTMRatFMR, roc_curve, calculate_eer

from .verification_utils import get_verification_pairs, getTMRatFMR, roc_curve
