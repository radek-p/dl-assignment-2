import tensorflow as tf


class ModelBuilder(object):
    def __init__(self, IMAGE_SIZE):
        self.summaries = []
        self.ORIGINAL_SIZE = IMAGE_SIZE
        self.IMAGE_SIZE = IMAGE_SIZE
        self.IMAGE_SIZE += (-IMAGE_SIZE) % 16

    @staticmethod
    def print_layer_info(depth, fst, snd, mlen=45):
        fst = "|   " * depth + str(fst)
        print(fst + " " * max(mlen - len(fst), 0) + str(snd))

    def layer_conv(self, signal, depth, in_fmaps, out_fmaps, kernel_size, include_relu=True, is_training=True):
        W = tf.get_variable(
            "W", [kernel_size, kernel_size, in_fmaps, out_fmaps], tf.float32,
            tf.truncated_normal_initializer(stddev=0.1)
        )

        self.summaries.append(tf.summary.histogram("conv_{}_W".format(depth), W))

        signal = tf.nn.conv2d(signal, W, [1, 1, 1, 1], "SAME", name="up_conv_conv2d")
        tf.layers.batch_normalization(signal, 3, training=is_training)

        suffix = ""
        if include_relu:
            signal = tf.nn.relu(signal)
            suffix = " + ReLU"

        self.print_layer_info(depth, "conv({} --> {}, {}){}".format(in_fmaps, out_fmaps, kernel_size, suffix),
                              signal.shape)
        return signal

    def layer_up_conv(self, signal, depth, is_training=True):
        batch_size = int(signal.shape[0])
        in_fmaps = int(signal.shape[3])
        out_fmaps = in_fmaps // 2
        target_h = int(signal.shape[1]) * 2
        target_w = int(signal.shape[2]) * 2

        W = tf.get_variable(
            "W", [2, 2, out_fmaps, in_fmaps], tf.float32,
            tf.truncated_normal_initializer(stddev=0.1)
        )

        self.summaries.append(tf.summary.histogram("upconv_{}_W".format(depth), W))

        signal = tf.nn.conv2d_transpose(
            signal,
            W,
            [batch_size, target_h, target_w, out_fmaps],
            [1, 2, 2, 1],
            name="upconv"
        )
        signal = tf.layers.batch_normalization(signal, 3, training=is_training)
        signal = tf.nn.relu(signal)

        self.print_layer_info(depth, "v   up conv(2x)", signal.shape)
        return signal

    def layer_max_pool_2x2(self, signal, depth):
        with tf.name_scope("max_pool"):
            signal = tf.nn.max_pool(signal, [1, 2, 2, 1], [1, 2, 2, 1], "SAME", name="max_pool")
            self.print_layer_info(depth, "|   max pool()", signal.shape)
            return signal

    def create_u_level(self, levels, signal, in_fmaps, out_fmaps, is_training):
        depth = levels[0]
        with tf.variable_scope("u_{}_conv1".format(depth)):
            signal = self.layer_conv(signal, depth, in_fmaps, out_fmaps, 3, is_training)
        with tf.variable_scope("u_{}_conv2".format(depth)):
            signal = self.layer_conv(signal, depth, out_fmaps, out_fmaps, 3, is_training)

        skip_connection = signal

        if len(levels) > 1:  # Not the last layer
            signal = self.layer_max_pool_2x2(signal, depth)
            signal = self.create_u_level(levels[1:], signal, out_fmaps, out_fmaps * 2, is_training)

            with tf.variable_scope("u_{}_upconv".format(depth)):
                signal = self.layer_up_conv(signal, depth)

            signal = tf.concat([skip_connection, signal], 3, "concat_skip_connection")
            self.print_layer_info(depth, "stack", signal.shape)

            with tf.variable_scope("u_{}_conv3".format(depth)):
                signal = self.layer_conv(signal, depth, 2 * out_fmaps, out_fmaps, 3, is_training)
            with tf.variable_scope("u_{}_conv4".format(depth)):
                signal = self.layer_conv(signal, depth, out_fmaps, out_fmaps, 3, is_training)

        return signal

    def create_model(self, signal, is_training):
        print("Creating the model")

        with tf.variable_scope("U_net_model"):
            signal = tf.image.resize_image_with_crop_or_pad(signal, self.IMAGE_SIZE, self.IMAGE_SIZE)
            self.summaries.append(tf.summary.image("input_image", signal))
            signal = self.create_u_level(range(5), signal, 3, 16, is_training)
            signal = self.layer_conv(signal, 0, 16, 3, 1, False)
            signal = tf.image.resize_image_with_crop_or_pad(signal, self.ORIGINAL_SIZE, self.ORIGINAL_SIZE)
            # output_image = tf.nn.softmax(signal)
            # tf.summary.image("output_image", output_image)

        return signal