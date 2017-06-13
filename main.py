# Autor:
# Radosław Piórkowski
# nr indeksu: 335451
import tensorflow as tf
import argparse
import sys
import os
import random
from tqdm import tqdm
import utility
import numpy as np
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
        self.validation_set_size += (-self.validation_set_size) % 16  # round the size to the multiple of 16
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


class Trainer(object):
    def __init__(self, tf_session, parameters=None):
        self.session = tf_session
        self.dataset = None
        self.m = {}
        self.model = self.m

        self.parameters = {
            "seed": -1,
            "validation_set_size": -1,
            "verbosity": 0,
            "training_steps": -1,
            "image_output_dir": "img_samples",
            "save_path_prefix": "./checkpoints",
            "start_from_step": 0,
            "run_idx": -1,
        }

        if parameters is not None:
            self.parameters.update(parameters)

        for parameter in self.parameters.keys():
            if self.parameters[parameter] is None or self.parameters[parameter] == -1:
                print("Missing value for parameter: '{}', program may not work correctly.".format(parameter))

        self.TRAINING_BATCH_SIZE = 32
        self.VALIDATION_BATCH_SIZE = 16
        self.INPUT_SIZE = 650
        self.IMAGE_SIZE = 325
        self.VALIDATION_STEPS = -1

        self.training_summaries = []
        self.validation_summaries = []

        self.parameters["image_output_dir"] = "img_samples_{}".format(self.parameters["run_idx"])
        if not os.path.exists(self.parameters["image_output_dir"]):
            os.makedirs(self.parameters["image_output_dir"])
            print(self.parameters["image_output_dir"] + " directory created.")

        print("Images will be shuffled with \t--seed={}".format(self.parameters["seed"]))
        self.generator = random.Random(self.parameters["seed"])

    def prepare_dataset(self, dataset_dir):
        print("Loading dataset")
        self.dataset = Spacenet2Dataset(dataset_dir, self.generator, self.parameters["validation_set_size"])
        self.dataset.load_dataset()
        self.VALIDATION_STEPS = len(self.dataset.get_validation_set_filenames()[0]) // self.VALIDATION_BATCH_SIZE

    def transform(self, transform_id, image):
        with tf.name_scope("image_transfrom"):
            num_rot = tf.mod(transform_id, 4, "rotation_idx")
            num_flip = tf.div(transform_id, 4)
            rotated = tf.image.rot90(image, num_rot, "rotation")

            not_flipped = rotated
            flipped = tf.image.flip_left_right(rotated)
            signal = tf.cond(tf.equal(num_flip, 1), lambda: flipped, lambda: not_flipped)

            signal.set_shape([self.IMAGE_SIZE, self.IMAGE_SIZE, 3])
            return signal

    def reverse_transform(self, transform_id, image):
        with tf.name_scope("image_transfrom"):
            full_rot = 4
            num_rot = full_rot - tf.mod(transform_id, 4, "rotation_idx")
            num_flip = tf.div(transform_id, 4)

            not_flipped = image
            flipped = tf.image.flip_left_right(image)
            signal = tf.cond(tf.equal(num_flip, 1), lambda: flipped, lambda: not_flipped)

            signal = tf.image.rot90(signal, num_rot, "rotation")

            signal.set_shape([self.IMAGE_SIZE, self.IMAGE_SIZE, 3])
            return signal

    @staticmethod
    def preprocess_image(signal):
        b, g, r = tf.unstack(signal, 3, 2)
        signal = tf.stack([r, g, b], 2)
        return signal

    def load_image(self, image_name):
        image = tf.read_file(image_name)
        image = tf.image.decode_jpeg(image, name="jpeg_decoding_2")
        image.set_shape([self.INPUT_SIZE, self.INPUT_SIZE, 3])
        image = tf.cast(image, tf.float32) / 255.
        image = tf.reshape(image, [1, self.INPUT_SIZE, self.INPUT_SIZE, 3])
        image = tf.image.resize_bilinear(image, size=[self.IMAGE_SIZE, self.IMAGE_SIZE])
        image = tf.reshape(image, [self.IMAGE_SIZE, self.IMAGE_SIZE, 3])
        return image

    def load_images(self, filename_queue):
        with tf.name_scope("image_loading"):
            image_name, heatmap_name = filename_queue
            image = self.load_image(image_name)
            heatmap = self.load_image(heatmap_name)
            return image, heatmap

    def create_training_pipeline(self, models_scope):
        with tf.name_scope("training_input_pipeline"):
            t_x = tf.Variable(tf.zeros([self.TRAINING_BATCH_SIZE, self.IMAGE_SIZE, self.IMAGE_SIZE, 3]),
                              trainable=False, name="t_x")
            t_y = tf.Variable(tf.zeros([self.TRAINING_BATCH_SIZE, self.IMAGE_SIZE, self.IMAGE_SIZE, 3]),
                              trainable=False, name="t_y")

            t_tf_type = tf.random_uniform([], 0, 8, dtype=tf.int32, name="t_tf_type")
            t_file_names = tf.train.slice_input_producer(self.dataset.get_training_set_filenames())

            t_image, t_heatmap = self.load_images(t_file_names)
            t_image = self.transform(t_tf_type, t_image)
            t_image = self.preprocess_image(t_image)
            t_heatmap = self.transform(t_tf_type, t_heatmap)

            t_x_batch, t_y_batch = tf.train.shuffle_batch_join([(t_image, t_heatmap)] * 24, self.TRAINING_BATCH_SIZE,
                                                               512, 32, name="t_shuffle_batch")

            t_load_next_batch_op = tf.group(
                tf.assign(t_x, t_x_batch),
                tf.assign(t_y, t_y_batch)
            )

        with tf.variable_scope(models_scope):
            signal = t_x
            b = utility.ModelBuilder(self.IMAGE_SIZE)
            signal = b.create_model(signal, True)
            self.training_summaries += b.summaries
            t_output_image = tf.nn.softmax(signal)

        with tf.name_scope("t_optimisation"):
            t_error_map = tf.nn.softmax_cross_entropy_with_logits(labels=t_y, logits=signal, name="softmax_loss")
            t_d0 = t_y[:, :, :, 1]
            t_d0 = tf.reshape(t_d0, [self.TRAINING_BATCH_SIZE, self.IMAGE_SIZE, self.IMAGE_SIZE, 1])
            t_d1 = tf.nn.dilation2d(t_d0, tf.ones([3, 3, 1]), [1, 1, 1, 1], [1, 1, 1, 1], "SAME")
            t_d2 = tf.nn.dilation2d(t_d1, tf.ones([3, 3, 1]), [1, 1, 1, 1], [1, 1, 1, 1], "SAME")
            t_d3 = tf.nn.dilation2d(t_d2, tf.ones([3, 3, 1]), [1, 1, 1, 1], [1, 1, 1, 1], "SAME")
            t_d3 = t_d3[:, :, :, 0]
            t_d3_normalized = t_d3 / tf.reduce_max(t_d3)
            t_error_severeness = tf.ones(t_error_map.get_shape()) + t_d3 * 40
            t_error_map_augmented = t_error_map * t_error_severeness
            t_error_map_normalized = t_error_map / tf.reduce_max(t_error_map)
            t_error_map_augmented_normalized = t_error_map_augmented / tf.reduce_max(t_error_map_augmented)
            t_loss = tf.reduce_mean(t_error_map)
            t_loss2 = tf.reduce_mean(t_error_map_augmented)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(0.001, epsilon=1e-4)
                t_optimize_op = optimizer.minimize(t_loss, name="t_optimize_op")
                t_optimize_op2 = optimizer.minimize(t_loss2, name="t_optimize_op2")

        self.training_summaries += [
            tf.summary.scalar("t_loss", t_loss),
            tf.summary.scalar("t_loss", t_loss2),
        ]

        return locals()

    def create_validation_pipeline(self, models_scope):
        with tf.name_scope("validation_input_pipeline"):
            v_img_shape = [self.VALIDATION_BATCH_SIZE, self.IMAGE_SIZE, self.IMAGE_SIZE, 3]
            v_x = tf.Variable(tf.zeros(v_img_shape), trainable=False, name="v_x")
            v_y = tf.Variable(tf.zeros(v_img_shape), trainable=False, name="v_y")
            v_tf_average_result = tf.Variable(tf.zeros(v_img_shape), trainable=False, name="v_tf_average_result")
            v_tf_average_result_sums = tf.reduce_sum(v_tf_average_result, axis=3)
            v_x_names = tf.Variable(tf.constant("$", tf.string, [self.VALIDATION_BATCH_SIZE]), trainable=False,
                                    name="v_x_names")
            v_y_names = tf.Variable(tf.constant("$", tf.string, [self.VALIDATION_BATCH_SIZE]), trainable=False,
                                    name="v_y_names")

            v_tf_num = tf.Variable(tf.zeros([]), trainable=False, name="v_tf_num")
            v_tf_type = tf.placeholder_with_default(tf.constant(0, dtype=tf.int32), [], "v_tf_type")

            v_file_names = tf.train.slice_input_producer(self.dataset.get_validation_set_filenames(), shuffle=False)
            v_x1_name, v_y1_name = v_file_names
            v_x1, v_y1 = self.load_images(v_file_names)
            v_x1 = self.preprocess_image(v_x1)

            v_x_batch, v_y_batch, v_x_batch_names, v_y_batch_names = tf.train.batch_join(
                [(v_x1, v_y1, v_x1_name, v_y1_name)] * 20, self.VALIDATION_BATCH_SIZE, 256
            )

        with tf.variable_scope(models_scope):
            signal = v_x
            signal = tf.map_fn(lambda signal_slice: self.transform(v_tf_type, signal_slice), signal)

            v_transformed_input = signal

            b = utility.ModelBuilder(self.IMAGE_SIZE)
            signal = b.create_model(signal, False)
            self.validation_summaries += b.summaries

            signal = tf.map_fn(lambda signal_slice: self.reverse_transform(v_tf_type, signal_slice), signal)
            v_tf_single_result = tf.nn.softmax(signal)

            # v_reverse_transformed_result = signal

            v_mean_result = tf.divide(v_tf_average_result, v_tf_num)
            # fix error map -- should be divided by max sum, not by 8
            v_error_map = tf.reduce_sum(-(v_y * tf.log(tf.clip_by_value(v_mean_result, 1e-10, 1.0))), axis=3)
            v_error_map_normalized = v_error_map / tf.reduce_max(v_error_map)
            v_average_output_loss = tf.reduce_mean(v_error_map)

            self.validation_summaries += [
                # tf.summary.image("v_transformed_input", v_transformed_input),
                # tf.summary.image("v_transformed_result", v_transformed_result),
                # tf.summary.image("v_tf_average_result", v_tf_average_result),
                tf.summary.scalar("v_average_output_loss", v_average_output_loss),
            ]

        with tf.name_scope("validation_input_pipeline_update_ops"):
            v_load_next_batch = tf.group(
                tf.assign(v_x, v_x_batch),
                tf.assign(v_y, v_y_batch),
                tf.assign(v_x_names, v_x_batch_names),
                tf.assign(v_y_names, v_y_batch_names)
            )

            v_add_partial_result = tf.group(
                tf.assign(
                    v_tf_average_result,
                    tf.add(v_tf_average_result, v_tf_single_result)
                ),
                tf.assign(v_tf_num, v_tf_num + 1)
            )

            v_reset_average_result = tf.group(
                tf.assign(
                    v_tf_average_result,
                    tf.tile(tf.constant(0., shape=[1, 1, 1, 1]), v_img_shape)
                ),
                tf.assign(v_tf_num, 0)
            )

        return locals()

    def create_graph(self):
        with tf.variable_scope("models") as models_scope_for_reusing:
            pass

        m_training = self.create_training_pipeline(models_scope_for_reusing)
        models_scope_for_reusing.reuse_variables()
        m_validation = self.create_validation_pipeline(models_scope_for_reusing)

        with tf.name_scope("summaries"):
            t_summary_writer = tf.summary.FileWriter("./logs/training_{}".format(self.parameters["run_idx"]))
            v_summary_writer = tf.summary.FileWriter("./logs/validation_{}".format(self.parameters["run_idx"]))

            for sw in [t_summary_writer, v_summary_writer]:
                sw.add_graph(self.session.graph)
                sw.flush()

            t_merged_summaries = tf.summary.merge(self.training_summaries)
            v_merged_summaries = tf.summary.merge(self.validation_summaries)

        with tf.name_scope("utilities"):
            saver = tf.train.Saver()

        self.model.update(locals())
        self.model.update(m_training)
        self.model.update(m_validation)

    def train_model(self):
        print("Training the model")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.session)
        m = self.m

        steps = self.parameters["training_steps"]
        small_steps = 2000
        big_steps = steps // small_steps
        offset = self.parameters["start_from_step"]

        self.validate_model(-1)

        for big_step in range(big_steps):
            for small_step in tqdm(range(small_steps), total=small_steps, desc=("epoch {}/{} ".format(big_step, big_steps)), leave=True):
                step = offset + big_step * small_steps + small_step

                self.session.run(m["t_load_next_batch_op"])

                _, t_merged_summaries, loss1, loss2 = self.session.run(
                    fetches=[
                        m["t_optimize_op"], m["t_merged_summaries"], m["t_loss"], m["t_loss2"],
                    ],
                )

                if small_step % 200 == 0:
                    images = self.session.run([m["t_x"], m["t_y"], m["t_output_image"], m["t_error_map_normalized"],
                                               m["t_error_map_augmented_normalized"], m["t_d3_normalized"]])
                    self.output_batch_stats(*images, label="t_b{:04d}_s{:04d}".format(big_step, small_step))

                if small_step % 10 == 0:
                    tqdm.write("loss: {},\t{}".format(loss1, loss2))

                m["t_summary_writer"].add_summary(t_merged_summaries, step)
                m["t_summary_writer"].flush()

            self.save_trained_values("big-step")
            self.validate_model(big_step * big_steps)
            tqdm.write("Big step #{} done.".format(big_step))

        coord.request_stop()
        coord.join(threads)

    def validate_model(self, training_step):
        m = self.m
        loss_sum = 0.

        for v_step in tqdm(range(self.VALIDATION_STEPS), total=self.VALIDATION_STEPS):
            tqdm.write("validation step: {}".format(v_step))

            self.session.run(m["v_reset_average_result"])
            self.session.run(m["v_load_next_batch"])

            for transformation_type in range(8):
                _, v_merged_summaries = self.session.run(
                    [
                        m["v_add_partial_result"],
                        m["v_merged_summaries"],
                    ],
                    feed_dict={
                        m["v_tf_type"]: transformation_type,
                    }
                )

                if v_step % 100 == 0:
                    images = self.session.run(
                        fetches=[
                            m["v_x"], m["v_y"], m["v_transformed_input"], m["v_tf_single_result"], m["v_mean_result"],
                            m["v_error_map_normalized"],
                        ],
                        feed_dict={
                            m["v_tf_type"]: transformation_type,
                        }
                    )
                    self.output_batch_stats(
                        *images,
                        label="v_r{:02d}_t{:04d}_s{:04d}_t{:04d}".format(
                            self.parameters["run_idx"], training_step, v_step, transformation_type
                        )
                    )

            loss, = self.session.run([
                m["v_average_output_loss"],
            ])
            loss_sum += loss
            tqdm.write("===> LOSS = {}, VAVLOSS = {}".format(loss, loss_sum / (v_step + 1)))
            tqdm.write("TX DONE")

        tqdm.write("TRANSFORM DONE")

        average_validation_loss = loss_sum / self.VALIDATION_STEPS
        summary = tf.Summary(value=[tf.Summary.Value(tag="average_validation_loss", simple_value=average_validation_loss)])
        m["v_summary_writer"].add_summary(summary, global_step=training_step)
        m["v_summary_writer"].flush()

    def output_batch_stats(self, *args, label):
        cols = [np.clip(np.concatenate(col, 0) * 255, 0, 255).astype(np.uint8) for col in args]
        for i in range(len(cols)):
            # jeśli obrazek jest 1-kanałowy, przekształcamy go na 3-kanałowy
            if len(cols[i].shape) == 2:
                cols[i] = np.reshape(cols[i], cols[i].shape + (1,))
                cols[i] = np.tile(cols[i], [1, 1, 3])
        img = Image.fromarray(np.concatenate(cols, 1))
        img.save("{}/{}.png".format(self.parameters["image_output_dir"], label))

    def save_trained_values(self, name):
        save_path = self.model["saver"].save(self.session,
                                             '{}/{}.ckpt'.format(self.parameters["save_path_prefix"], name))
        print("Model values saved: {}".format(save_path))

    def load_trained_values(self, name):
        checkpoint_path = '{}/{}.ckpt'.format(self.parameters["save_path_prefix"], name)
        self.model["saver"].restore(self.session, checkpoint_path)
        print("Model values restored from checkpoint: {}".format(checkpoint_path))

    def init_values(self, checkpoint):
        if checkpoint == "":
            self.session.run(tf.global_variables_initializer())
        else:
            self.load_trained_values(checkpoint)


def main(argv):
    parser = argparse.ArgumentParser(prog='main.py')
    parser.add_argument("-d", "--dataset", required=True, help="Directory containing spacenet2 dataset.")
    parser.add_argument("-v", "--verbose", required=False, action="store_true", help="Turn on verbose logs.")
    parser.add_argument("--debug", required=False, action="store_true", help="Turn on TF Debugger.")
    parser.add_argument("--seed", required=False, default=random.randint(0, sys.maxsize), type=int,
                        help="Set seed for pseudo-random shuffle of data.")
    parser.add_argument("--start-from-step", required=False, default=0, type=int,
                        help="Idx of first step -- used when state was saved to checkpoint to ensure consistent "
                             "numbering.")
    parser.add_argument("--training-steps", required=False, default=50000, type=int,
                        help="Number of training steps.")

    def validation_set_size_type(ratio_str):
        ratio = float(ratio_str)
        if ratio <= 0.0 or ratio >= 1.0:
            raise argparse.ArgumentTypeError("{} not in range (0.0, 1.0)".format(ratio))
        return ratio

    parser.add_argument("--validation-set-size", required=False, default=0.10, type=validation_set_size_type,
                        help="fraction of images to use during validation")
    parser.add_argument("--start-from-checkpoint", required=False, default="",
                        help="checkpoint location")

    options = parser.parse_args(argv)

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
            "start_from_step": options.start_from_step,
            "run_idx": get_run_idx(),
        })

        t.prepare_dataset(dataset_dir=options.dataset)
        print("Creating model")
        t.create_graph()
        t.init_values(options.start_from_checkpoint)
        try:
            t.train_model()
        except KeyboardInterrupt:
            print("Interrupted, saving")
            t.save_trained_values("interrupt")


def get_run_idx():
    idx = 0
    while os.path.exists("logs/training_{}".format(idx)):
        print("EXISTS logs/training_{}".format(idx))
        idx += 1

    print("choosing logs/training_{}".format(idx))
    return idx


if __name__ == '__main__':
    main(sys.argv[1:])
