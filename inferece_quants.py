from lite.lite_wrapper import YOLOv8LiteWrapper
import tensorflow as tf
import warnings
tf.experimental.numpy.experimental_enable_numpy_behavior()
from profilers import ModelsProfiler, InferenceProfiler
shuffle_lite_model = YOLOv8LiteWrapper("quantized_models/shuffle.tflite")
ghost_lite_model = YOLOv8LiteWrapper("quantized_models/ghost.tflite")
csp_lite_model = YOLOv8LiteWrapper("quantized_models/csp.tflite")

warnings.filterwarnings("ignore")

profilers = [
    # COCOMetricsCalculatorLite(val_ds),
    InferenceProfiler(repeats=50, device='CPU:0', warmup_steps=10, batch_timing=False),
]

models = {
    'ShuffleNetV2-YOLOv8': shuffle_lite_model,
    'CSPDarkNet-YOLOv8': csp_lite_model,
    'GhostNetv2-YOLOv8': ghost_lite_model,
}
input_data = tf.zeros([1, 640, 640, 3])
general_profiles = ModelsProfiler(profilers=profilers)
profile_results = general_profiles.profile(models, tf.cast(input_data, tf.float16))
print(profile_results[['fps']])
