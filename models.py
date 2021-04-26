import tensorflow as tf

resnet50 = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1)

# models = {'resnet50_2D' : ResNet502D,
#           'resnet50_3D' : ResNet503D,
#           'resnet50_3D_no_BN' : ResNet503D_no_batchnorm}
