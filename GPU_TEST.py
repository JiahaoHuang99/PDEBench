import torch

flag = torch.cuda.is_available()
print(flag)

ngpu = 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3, 3).cuda())



import tensorflow as tf

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check if TensorFlow was built with CUDA (GPU support)
print("Built with CUDA:", tf.test.is_built_with_cuda())

# Check if a GPU is available and TensorFlow is using it
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU Devices: ", gpus)
else:
    print("No GPU found")

# Optional: Perform a simple operation using TensorFlow
if gpus:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
    print("Simple TensorFlow operation result:\n", c)
