import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

from csp_darknet import create_xs_ghostnet_yolov8
from ghostnet_v2 import create_0_5_ghostnet_yolov8
from shufflenet_v2 import create_0_5_shufflenet_yolov8


def get_flops(model, input_shape=(1, 640, 640, 3)):
    inputs = tf.random.normal(input_shape)

    @tf.function
    def model_fn(x):
        return model(x)

    concrete = model_fn.get_concrete_function(inputs)

    flop_profile = profile(
        concrete.graph,
        options=ProfileOptionBuilder.float_operation()
    )
    return flop_profile.total_float_ops


# Example usage
model_shuffle = create_0_5_shufflenet_yolov8(2)
csp_darknet_yolo = create_xs_ghostnet_yolov8(2)
ghost_model = create_0_5_ghostnet_yolov8(2)


csp_darknet_yolo.load_weights("best_weights/best_csp_darknet_yolo(1).keras")
model_shuffle.load_weights("best_weights/best_shuffle_yolo.keras")
ghost_model.load_weights("best_weights/best_ghost_yolo.keras")


flops = get_flops(model_shuffle)
print(f"FLOPs: {flops:,}")
print(f"GFLOPs: {flops / 1e9:.3f}")

flops = get_flops(csp_darknet_yolo)
print(f"FLOPs: {flops:,}")
print(f"GFLOPs: {flops / 1e9:.3f}")

flops = get_flops(ghost_model)
print(f"FLOPs: {flops:,}")
print(f"GFLOPs: {flops / 1e9:.3f}")