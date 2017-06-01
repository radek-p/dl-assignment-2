# Autor:
# Radosław Piórkowski
# nr indeksu: 335451
from time import sleep

import numpy as np
import tensorflow as tf
import argparse
import sys
import os
import random
from tqdm import tqdm
from PIL import Image


class Spacenet2Dataset(object):
    def __init__(self, dataset_dir, generator, validation_set_size_ratio):
        self.generator = generator
        self.validation_set_size_ratio = validation_set_size_ratio
        self.dataset_dir = dataset_dir

        self.training_images = None
        self.training_heatmaps = None
        self.validation_images = None
        self.validation_heatmaps = None

        self.dataset_size = -1
        self.training_set_size = -1
        self.validation_set_size = -1

    def load_dataset(self):
        images_folder = os.path.join(self.dataset_dir, "images")
        heatmaps_folder = os.path.join(self.dataset_dir, "heatmaps")

        try:
            image_names = os.listdir(images_folder)
        except FileNotFoundError:
            print("Cannot find images in '{}', exiting.".format(images_folder))
            sys.exit(1)

        try:
            heatmap_names = os.listdir(heatmaps_folder)
        except FileNotFoundError:
            print("Cannot find heatmaps in '{}', exiting.".format(images_folder))
            sys.exit(1)

        if len(image_names) != len(heatmap_names):
            print("Broken dataset: different number of images and heatmaps ({} != {}).".format(len(image_names),
                                                                                               len(heatmap_names)))
            sys.exit(2)

        for name in image_names:
            if name not in heatmap_names:
                print("Broken dataset: missing heatmap for image '{}'".format(name))
                sys.exit(2)

        print("Dataset Ok")

        image_names.sort()
        self.generator.shuffle(image_names)

        self.dataset_size = len(image_names)
        self.validation_set_size = int(self.dataset_size * self.validation_set_size_ratio)
        self.training_set_size = self.dataset_size - self.validation_set_size

        training_image_names = image_names[:self.training_set_size]
        validation_image_names = image_names[self.training_set_size:]

        # if self.parameters["verbosity"] > 0:
        print("train set size", self.training_set_size)
        print("validation set size", self.validation_set_size)

        print("train set[:5]", training_image_names[:5])
        print("validation set[:5]", validation_image_names[:5])

        print("Opening images:")
        print("--> training images")
        self.training_images = self._open_dataset(images_folder, training_image_names)
        print("--> training heatmaps")
        self.training_heatmaps = self._open_dataset(heatmaps_folder, training_image_names)
        print("--> validation images")
        self.validation_images = self._open_dataset(images_folder, validation_image_names)
        print("--> validation heatmaps")
        self.validation_heatmaps = self._open_dataset(heatmaps_folder, validation_image_names)

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
        }
        if parameters is not None:
            self.parameters.update(parameters)

        for parameter in self.parameters.keys():
            if self.parameters[parameter] is None or self.parameters[parameter] == -1:
                print("Missing value for parameter: '{}', program may not work correctly.".format(parameter))

        print("Images will be shuffled with \t--seed={}".format(self.parameters["seed"]))
        self.generator = random.Random(self.parameters["seed"])

    def prepare_dataset(self, dataset_dir):
        self.dataset = Spacenet2Dataset(dataset_dir, self.generator, self.parameters["validation_set_size"])
        self.dataset.load_dataset()

    def print_layer_info(self, depth, fst, snd, mlen=45):
        fst = "|   " * depth + str(fst)
        print(fst + " " * max(mlen - len(fst), 0) + str(snd))

    def layer_conv(self, signal, depth, in_fmaps, out_fmaps, kernel_size, include_relu=True):
        with tf.name_scope("conv_{}".format(depth)):
            W = tf.Variable(
                tf.truncated_normal([kernel_size, kernel_size, in_fmaps, out_fmaps], stddev=0.1, dtype=self.float),
                name="W")
            b = tf.Variable(tf.truncated_normal([out_fmaps], stddev=0.1, dtype=self.float), name="b")

            tf.summary.histogram("conv_{}_W".format(depth), W)
            tf.summary.histogram("conv_{}_b".format(depth), b)

            signal = tf.nn.conv2d(signal, W, [1, 1, 1, 1], "SAME", name="up_conv_conv2d") + b

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

            signal = tf.image.resize_bilinear(signal, [target_h, target_w], name="up_conv_resize_bilinear")
            signal = self.layer_conv(signal, depth + 1, in_fmaps, out_fmaps, 3, False)

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
            with tf.name_scope("u_{}_step".format(depth)):
                signal = self.layer_max_pool_2x2(signal, depth)
                signal = self.create_u_level(levels[1:], signal, out_fmaps, out_fmaps * 2)
                signal = self.layer_up_conv(signal, depth)

            with tf.name_scope("u_{}_end".format(depth)):
                signal = tf.concat([skip_connection, signal], 3, "concat_skip_connection")
                self.print_layer_info(depth, "stack", signal.shape)
                signal = self.layer_conv(signal, depth, 2 * out_fmaps, out_fmaps, 3)
                signal = self.layer_conv(signal, depth, out_fmaps, out_fmaps, 3)

        return signal

    def create_model(self):
        print("Creating the model")
        input_h, input_w = 325, 325
        target_h, target_w = 336, 336

        with tf.name_scope("placeholders"):
            x = tf.placeholder(dtype=self.float, shape=[None, input_h, input_w, 3], name="x")
            target_y = tf.placeholder(dtype=self.float, shape=[None, input_h, input_w, 3], name="target_y")

        with tf.name_scope("main_graph"):
            signal = tf.image.resize_image_with_crop_or_pad(x, target_h, target_w)
            signal = self.create_u_level(range(5), signal, 3, 32)
            signal = self.layer_conv(signal, 0, 32, 3, 1)
            signal = tf.image.resize_image_with_crop_or_pad(x, input_h, input_w)

        with tf.name_scope("metrics"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_y, logits=signal, name="softmax_loss")
            tf.summary.scalar("loss", loss)

        with tf.name_scope("utilities"):
            saver = tf.train.Saver()

        with tf.name_scope("summaries"):
            summary_writer = tf.summary.FileWriter("./logs/summaries")
            merged_summaries = tf.summary.merge_all()

        self.model.update(locals())

    def train_model(self):
        print("Training the model")
        self.session.run(tf.global_variables_initializer())
        m = self.m

        steps = self.parameters["training_steps"]
        small_steps = 100
        big_steps = steps // small_steps

        for big_step in tqdm(range(big_steps), desc="epochs"):
            # print("TEST", end="") # print here like this
            for step in tqdm(range(small_steps), total=small_steps, desc="steps ", leave=True):
                # x, y = self.input_data.train.next_batch(self.parameters["batch_size"])
                # x = self.preprocess_input(x)
                sleep(0.1)
                # merged_summaries = self.session.run(
                #     fetches=[
                #         m["merged_summaries"],
                #     ],
                #     feed_dict={
                #     }
                # )
                #
                # m["summary_writer"].add_summary(merged_summaries, step)
                pass

            self.train_model__report_stats(big_step)

            # print(self.training_images[513, 13, 18, :])
            # print(self.training_heatmaps[514, 13, 18, :])
            # print(self.validation_images[314, 13, 18, :])
            # print(self.validation_heatmaps[314, 13, 18, :])

    def train_model__report_stats(self, big_step):
        accuracy = self.session.run(
            fetches=[
            ],
            feed_dict={
            }
        )
        print()
        print("Big step summary: {}".format(42))
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

    with tf.Session() as session:
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
        print("Loading dataset")

        t.prepare_dataset(dataset_dir=options.dataset)
        print("Creating model")
        t.create_model()
        t.train_model()


if __name__ == '__main__':
    main(sys.argv[1:])