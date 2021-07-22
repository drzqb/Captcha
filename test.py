import tensorflow as tf

resnet=tf.keras.applications.resnet50.ResNet50(include_top=False,input_shape=[200,50,3])
resnet.summary(line_length=300)