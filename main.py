# Autor:
# Radosław Piórkowski
# nr indeksu: 335451
from time import sleep

import math
import numpy as np
import select
import tensorflow as tf
import argparse
import sys
import os
import random
from tensorflow.python.training.adam import AdamOptimizer
from tqdm import tqdm
from PIL import Image


class Spacenet2Dataset(object):
    def __init__(self, dataset_dir, generator, validation_set_size_ratio):
        self.generator = generator
        self.validation_set_size_ratio = validation_set_size_ratio
        self.dataset_dir = dataset_dir

        self.images_folder = os.path.join(self.dataset_dir, "images")
        self.heatmaps_folder = os.path.join(self.dataset_dir, "heatmaps")

        self.training_image_names = None
        self.validation_image_names = None
        self.training_heatmap_names = None
        self.validation_heatmap_names = None

        self.training_images = None
        self.training_heatmaps = None
        self.validation_images = None
        self.validation_heatmaps = None

        self.dataset_size = -1
        self.training_set_size = -1
        self.validation_set_size = -1

    def load_dataset(self):
        try:
            image_names = os.listdir(self.images_folder)
        except FileNotFoundError:
            print("Cannot find images in '{}', exiting.".format(self.images_folder))
            sys.exit(1)

        try:
            heatmap_names = os.listdir(self.heatmaps_folder)
        except FileNotFoundError:
            print("Cannot find heatmaps in '{}', exiting.".format(self.heatmaps_folder))
            sys.exit(1)

        if len(image_names) != len(heatmap_names):
            print("Broken dataset: different number of images and heatmaps ({} != {}).".format(len(image_names),
                                                                                               len(heatmap_names)))
            sys.exit(2)

        for name in image_names:
            if name not in heatmap_names:
                print("Broken dataset: missing heatmap for image '{}'".format(name))
                sys.exit(2)

        # here image_names and heatmap_names are identical
        names = image_names
        names.sort()

        name_pairs = [
            (os.path.join(self.images_folder, filename), os.path.join(self.heatmaps_folder, filename))
            for filename in names
        ]

        print("Dataset Ok")
        print(name_pairs[:5])

        self.generator.shuffle(name_pairs)
        image_names, heatmap_names = zip(*name_pairs)

        self.dataset_size = len(image_names)
        self.validation_set_size = int(self.dataset_size * self.validation_set_size_ratio)
        self.training_set_size = self.dataset_size - self.validation_set_size

        self.training_image_names = image_names[:self.training_set_size]
        self.validation_image_names = image_names[self.training_set_size:]
        self.training_heatmap_names = heatmap_names[:self.training_set_size]
        self.validation_heatmap_names = heatmap_names[self.training_set_size:]

        # if self.parameters["verbosity"] > 0:
        print("train set size", self.training_set_size)
        print("validation set size", self.validation_set_size)

        print("train set[:5]", self.training_image_names[:5])
        print("train set[:5]", self.training_heatmap_names[:5])
        print("validation set[:5]", self.validation_image_names[:5])
        print("validation set[:5]", self.validation_heatmap_names[:5])

    def get_training_set_filenames(self):
        return self.training_image_names, self.training_heatmap_names

    def get_validation_set_filenames(self):
        return self.validation_image_names, self.validation_heatmap_names

    def _do_open_images(self):
        print("Opening images:")
        print("--> training images")
        self.training_images = self._open_dataset(self.images_folder, self.training_image_names)
        print("--> training heatmaps")
        self.training_heatmaps = self._open_dataset(self.heatmaps_folder, self.training_image_names)
        print("--> validation images")
        self.validation_images = self._open_dataset(self.images_folder, self.validation_image_names)
        print("--> validation heatmaps")
        self.validation_heatmaps = self._open_dataset(self.heatmaps_folder, self.validation_image_names)

    def _open_dataset(self, path_prefix, image_name_list):
        total = len(image_name_list)
        target_width, target_height = 325, 325
        res = np.zeros([total, 325, 325, 3], dtype=np.uint8)
        for i, name in tqdm(enumerate(image_name_list), total=total):
            with Image.open(os.path.join(path_prefix, name)) as img:
                res[i, :, :, :] = img.resize([target_width, target_height], Image.BILINEAR)
        return res


class Trainer(object):
    def __init__(self, tf_session, parameters=None):
        self.float = tf.float32
        self.session = tf_session
        self.dataset = None
        self.m = {}
        self.model = self.m

        self.parameters = {
            "seed": -1,
            "validation_set_size": -1,
            "verbosity": 0,
            "training_steps": -1,
            "input_size": 650,
            "input_small_size": 325,
        }
        if parameters is not None:
            self.parameters.update(parameters)

        for parameter in self.parameters.keys():
            if self.parameters[parameter] is None or self.parameters[parameter] == -1:
                print("Missing value for parameter: '{}', program may not work correctly.".format(parameter))

        print("Images will be shuffled with \t--seed={}".format(self.parameters["seed"]))
        self.generator = random.Random(self.parameters["seed"])

    def prepare_dataset(self, dataset_dir):
        print("Loading dataset")
        self.dataset = Spacenet2Dataset(dataset_dir, self.generator, self.parameters["validation_set_size"])
        self.dataset.load_dataset()

    def print_layer_info(self, depth, fst, snd, mlen=45):
        fst = "|   " * depth + str(fst)
        print(fst + " " * max(mlen - len(fst), 0) + str(snd))

    def layer_conv(self, signal, depth, in_fmaps, out_fmaps, kernel_size, include_relu=True):
        with tf.name_scope("conv_{}".format(depth)):
            # stddev = math.sqrt(2./float((kernel_size ^ 2) * in_fmaps))
            # stddev = 2. / (kernel_size * kernel_size * in)
            stddev = 0.1
            print("stddevs: {} vs. {}".format(stddev, 0.1))

            # tf.get_variable()
            W = tf.Variable(
                tf.truncated_normal([kernel_size, kernel_size, in_fmaps, out_fmaps], stddev=stddev, dtype=self.float),
                name="W")
            gamma = tf.Variable(tf.truncated_normal([in_fmaps], mean=1.0, stddev=0.1, dtype=self.float), name="b")
            beta = tf.Variable(tf.truncated_normal([in_fmaps], stddev=0.1, dtype=self.float), name="b")

            tf.summary.histogram("conv_{}_W".format(depth), W)
            tf.summary.histogram("conv_{}_gamma".format(depth), gamma)
            tf.summary.histogram("conv_{}_beta".format(depth), beta)

            mean, variance = tf.nn.moments(signal, [0, 1, 2])
            signal = tf.nn.batch_normalization(signal, mean, variance, beta, gamma, 1e-6, "batch_norm")
            signal = tf.nn.conv2d(signal, W, [1, 1, 1, 1], "SAME", name="up_conv_conv2d")

            suffix = ""
            if include_relu:
                signal = tf.nn.relu(signal)
                suffix = " + ReLU"

            self.print_layer_info(depth, "conv({} --> {}, {}){}".format(in_fmaps, out_fmaps, kernel_size, suffix),
                                  signal.shape)
            return signal

    def layer_up_conv(self, signal, depth):
        with tf.name_scope("up_conv"):
            in_fmaps = int(signal.shape[3])
            out_fmaps = in_fmaps // 2
            target_h = int(signal.shape[1]) * 2
            target_w = int(signal.shape[2]) * 2

            print("infmaps = {}, outfmaps = {}".format(in_fmaps, out_fmaps))

            stddev = 0.1
            W = tf.Variable(
                tf.truncated_normal([2, 2, out_fmaps, in_fmaps], stddev=stddev, dtype=self.float),
                name="W"
            )
            # b = tf.Variable(tf.truncated_normal([out_fmaps], stddev=0.1, dtype=self.float), name="b")
            gamma = tf.Variable(tf.truncated_normal([in_fmaps], mean=1.0, stddev=0.1, dtype=self.float), name="b")
            beta = tf.Variable(tf.truncated_normal([in_fmaps], stddev=0.1, dtype=self.float), name="b")
            # signal = tf.image.resize_bilinear(signal, [target_h, target_w], name="up_conv_resize_bilinear")
            # signal = self.layer_conv(signal, depth + 1, in_fmaps, out_fmaps, 3, True)
            mean, variance = tf.nn.moments(signal, [0, 1, 2])
            signal = tf.nn.batch_normalization(signal, mean, variance, beta, gamma, 1e-6, "batch_norm")

            signal = tf.nn.conv2d_transpose(
                signal,
                W,
                [int(signal.shape[0]), target_h, target_w, out_fmaps],
                [1, 2, 2, 1],
                name="upconv"
            )

            signal = tf.nn.relu(signal)
            self.print_layer_info(depth, "v   up conv(2x)", signal.shape)
            return signal

    def layer_max_pool_2x2(self, signal, depth):
        with tf.name_scope("max_pool"):
            signal = tf.nn.max_pool(signal, [1, 2, 2, 1], [1, 2, 2, 1], "SAME", name="max_pool")
            self.print_layer_info(depth, "|   max pool()", signal.shape)
            return signal

    def create_u_level(self, levels, signal, in_fmaps, out_fmaps):
        depth = levels[0]
        img_size = signal.shape[1]
        with tf.name_scope("u_{}_begin".format(depth)):
            signal = self.layer_conv(signal, depth, in_fmaps, out_fmaps, 3)
            signal = self.layer_conv(signal, depth, out_fmaps, out_fmaps, 3)
            skip_connection = signal

        if len(levels) > 1:  # Not the last layer
            with tf.name_scope("u_{}_step1".format(depth)):
                signal = self.layer_max_pool_2x2(signal, depth)

            signal = self.create_u_level(levels[1:], signal, out_fmaps, out_fmaps * 2)

            with tf.name_scope("u_{}_step2".format(depth)):
                signal = self.layer_up_conv(signal, depth)

            with tf.name_scope("u_{}_end".format(depth)):
                signal = tf.concat([skip_connection, signal], 3, "concat_skip_connection")
                self.print_layer_info(depth, "stack", signal.shape)
                signal = self.layer_conv(signal, depth, 2 * out_fmaps, out_fmaps, 3)
                signal = self.layer_conv(signal, depth, out_fmaps, out_fmaps, 3)
        return signal

    def transform(self, transform_id, image, is_image=False):
        with tf.name_scope("image_transfrom"):
            # overlay = tf.lin_space(0., 324., 325)
            # overlay = tf.reshape(overlay, [1, 325, 1])
            # overlay = tf.tile(overlay, [325,1,3])
            # print("OVERLAY SHAPE: {}\n\n\n".format(overlay.shape))
            # overlay1 = tf.sigmoid(10 - tf.square(overlay - tf.cast(10, tf.float32)))
            # overlay2 = tf.sigmoid(10 - tf.square(overlay - tf.cast(40 * transform_id, tf.float32)))
            # image = tf.maximum(image, overlay1)

            k = tf.mod(transform_id, 4, "rotation_idx")
            rotated = tf.image.rot90(image, k, "rotation")

            not_flipped = rotated
            flipped = tf.image.flip_left_right(rotated)
            signal = tf.cond(tf.equal(tf.div(transform_id, 4), 1), lambda: flipped, lambda: not_flipped)

            # signal = tf.maximum(overlay2, signal)
            # signal = tf.reshape(signal, [325, 325, 3])

            signal.set_shape([325, 325, 3])
            if is_image:
                b, g, r = tf.unstack(signal, 3, 2)
                signal = tf.stack([r, g, b], 2)
                # signal = tf.image.per_image_standardization(signal)
                signal *= 2.
            return signal

    def load_images(self, filename_queue):
        with tf.name_scope("image_loading"):
            image_name, heatmap_name = filename_queue

            image = tf.read_file(image_name)
            heatmap = tf.read_file(heatmap_name)

            image = tf.image.decode_jpeg(image, name="jpeg_decoding_1")
            image.set_shape([650, 650, 3])
            image = tf.cast(image, self.float) / 255.
            image = tf.reshape(image, [1, 650, 650, 3])
            image = tf.image.resize_bilinear(image, size=[325, 325])
            image = tf.reshape(image, [325, 325, 3])

            heatmap = tf.image.decode_jpeg(heatmap, name="jpeg_decoding_2")
            heatmap.set_shape([650, 650, 3])
            heatmap = tf.cast(heatmap, self.float) / 255.
            heatmap = tf.reshape(heatmap, [1, 650, 650, 3])
            heatmap = tf.image.resize_bilinear(heatmap, size=[325, 325])
            heatmap = tf.reshape(heatmap, [325, 325, 3])

            return image, heatmap

    def create_model(self):
        print("Creating the model")

        with tf.name_scope("input_pipeline"):
            transformation_type = tf.random_uniform([], 0, 8, dtype=tf.int32, name="random_transformation_type")
            filename_queue = tf.train.slice_input_producer(self.dataset.get_training_set_filenames())
            image, heatmap = self.load_images(filename_queue)
            image = self.transform(transformation_type, image, True)
            heatmap = self.transform(transformation_type, heatmap, False)
            x_batch, y_batch = tf.train.shuffle_batch_join([(image, heatmap)] * 24, 32, 512, 32,
                                                           name="shuffle_batch_join")

        with tf.name_scope("test_input_pipeline"):
            vx_name = tf.placeholder(tf.string, [], "vx_name")
            vy_name = tf.placeholder(tf.string, [], "vy_name")
            image, heatmap = self.load_images((vx_name, vy_name))
            images = [self.transform(transform_id, image, True) for transform_id in range(8)]
            heatmaps = [self.transform(transform_id, heatmap, False) for transform_id in range(8)]
            xv_batch = tf.stack(images, 0)
            yv_batch = tf.stack(heatmaps, 0)

        input_h, input_w = 325, 325
        target_h, target_w = 336, 336
        # input_h, input_w = 650, 650
        # target_h, target_w = 656, 656

        with tf.name_scope("placeholders"):
            # global_step = tf.Variable(0, dtype=tf.int32, name="global_step")

            #     x = tf.placeholder(dtype=self.float, shape=[None, input_h, input_w, 3], name="x")
            #     target_y = tf.placeholder(dtype=self.float, shape=[None, input_h, input_w, 3], name="target_y")
            pass

        x = x_batch
        target_y = y_batch
        signal = x

        with tf.name_scope("main_graph"):
            signal = tf.image.resize_image_with_crop_or_pad(signal, target_h, target_w)
            resized_input = signal
            tf.summary.image("input_image", signal)
            signal = self.create_u_level(range(5), signal, 3, 16)
            signal = self.layer_conv(signal, 0, 16, 3, 1, False)
            signal = tf.image.resize_image_with_crop_or_pad(signal, input_h, input_w)
            output_image = tf.nn.softmax(signal)
            tf.summary.image("output_image", output_image)
            tf.summary.image("target_y", target_y)

        with tf.name_scope("optimisation"):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=target_y, logits=signal, name="softmax_loss")
            )
            print(loss.shape)
            optimize_op = AdamOptimizer(0.001, name="Michal", epsilon=1e-4).minimize(loss)
            tf.summary.scalar("loss", loss)

        with tf.name_scope("summaries"):
            summary_writer = tf.summary.FileWriter("./logs/summaries")
            summary_writer.add_graph(self.session.graph)
            summary_writer.flush()
            # sys.exit(42)
            merged_summaries = tf.summary.merge_all()

        with tf.name_scope("utilities"):
            saver = tf.train.Saver()

        self.model.update(locals())

    def train_model(self):
        print("Training the model")
        self.session.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(self.session)
        m = self.m

        steps = self.parameters["training_steps"]
        small_steps = 100
        big_steps = steps // small_steps

        for big_step in range(big_steps):  # tqdm(range(big_steps), desc="epochs"):
            # print("TEST", end="") # print here like this
            # inner_range =
            for small_step in tqdm(range(small_steps), total=small_steps, desc="steps ", leave=True):
                _, merged_summaries, loss = self.session.run(
                    fetches=[
                        m["optimize_op"],
                        m["merged_summaries"],
                        m["loss"],
                    ],
                    feed_dict={
                        # m["global_step"]: big_step * big_steps + step
                    }
                )
                if select.select([sys.stdin, ], [], [], 0.0)[0]:
                    command = input()
                    if command == "t":
                        tqdm.write("WOULD RUN TESTS NOW")

                tqdm.write("loss: {}".format(loss))
                m["summary_writer"].add_summary(merged_summaries, big_step * big_steps + small_step)

            m["summary_writer"].flush()
            self.train_model__report_stats(big_step)

    def train_model__report_stats(self, big_step):
        accuracy = self.session.run(
            fetches=[
                # self.m["resized_input"]
            ],
            feed_dict={
            }
        )
        print()
        print("Big step #{} done.".format(big_step))
        print("-----------------------------\n")
        # print("\n[{}] accuracy: {}".format(big_step, accuracy))

    def save_trained_values(self, name):
        save_path = self.model["saver"].save(self.session,
                                             '{}/{}.ckpt'.format(self.parameters["save_path_prefix"], name))
        print("Model values saved: {}".format(save_path))

    def load_trained_values(self, name):
        checkpoint_path = '{}/{}.ckpt'.format(self.parameters["save_path_prefix"], name)
        self.model["saver"].restore(self.session, checkpoint_path)
        print("Model values restored from checkpoint: {}".format(checkpoint_path))


def main(argv):
    print("Hello.")

    parser = argparse.ArgumentParser(prog='main.py')
    parser.add_argument("-d", "--dataset", required=True, help="Directory containing spacenet2 dataset.")
    parser.add_argument("-v", "--verbose", required=False, action="store_true", help="Turn on verbose logs.")
    parser.add_argument("--debug", required=False, action="store_true", help="Turn on TF Debugger.")
    parser.add_argument("--seed", required=False, default=random.randint(0, sys.maxsize), type=int,
                        help="Set seed for pseudo-random shuffle of data.")
    parser.add_argument("--training-steps", required=False, default=10000, type=int,
                        help="Number of training steps.")

    def validation_set_size_type(ratio_str):
        ratio = float(ratio_str)
        if ratio <= 0.0 or ratio >= 1.0:
            raise argparse.ArgumentTypeError("{} not in range (0.0, 1.0)".format(ratio))
        return ratio

    parser.add_argument("--validation-set-size", required=False, default=0.2, type=validation_set_size_type,
                        help="fraction of images to use during validation")

    options = parser.parse_args(argv)
    print(options)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        if options.debug:
            print("Running in debug mode")
            from tensorflow.python import debug as tf_debug
            session = tf_debug.LocalCLIDebugWrapperSession(session)
            session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        t = Trainer(session, {
            "seed": int(options.seed),
            "validation_set_size": options.validation_set_size,
            "verbosity": 3 if options.verbose else 0,
            "training_steps": options.training_steps,
        })

        t.prepare_dataset(dataset_dir=options.dataset)
        print("Creating model")
        t.create_model()
        t.train_model()


if __name__ == '__main__':
    main(sys.argv[1:])
