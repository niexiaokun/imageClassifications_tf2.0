import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import defaultdict
import os
import cv2
import glob
import random


def get_data_from_origin():
    pass


def make_records(data_dir, out_dir, out_filename):
    label_paths = glob.glob(os.path.join(data_dir, "*txt"))
    random.seed(1234567)
    random.shuffle(label_paths)
    writer = tf.python_io.TFRecordWriter(os.path.join(out_dir, out_filename))
    idx = 0
    for label_path in label_paths:
        with open(label_path, 'r') as label_file:
            info = label_file.readline().split(', ')
        img_file = os.path.join(data_dir, info[0])
        with tf.gfile.FastGFile(img_file, 'rb') as gf:
            img_buf = gf.read()
        label = int(info[1])
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encode': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_buf])),
            'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['JPEG'.encode('utf-8')])),
            'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[info[0].encode('utf-8')])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        }))
        writer.write(example.SerializeToString())
        idx += 1
        print("{} convert {}".format(idx, img_file))


def random_distort_color(image):
    """list 4 ways to distort color, which will be random selected"""
    def distort_way_1(image):
        image = tf.image.random_brightness(image, max_delta=24. / 255.)
        image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
        # image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
        return image

    def distort_way_2(image):
        image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
        image = tf.image.random_brightness(image, max_delta=24. / 255.)
        image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
        # image = tf.image.random_hue(image, max_delta=0.2)
        return image

    def distort_way_3(image):
        image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
        # image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=24. / 255.)
        image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
        return image

    def distort_way_4(image):
        # image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
        image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
        image = tf.image.random_brightness(image, max_delta=24. / 255.)
        return image

    with tf.name_scope('distort_color', [image]):
        random_ordering = tf.random_uniform([], minval=1, maxval=5, dtype=tf.int32)
        image = tf.cond(tf.equal(random_ordering, 1),
                        lambda: distort_way_1(image), lambda: image)
        image = tf.cond(tf.equal(random_ordering, 2),
                        lambda: distort_way_2(image), lambda: image)
        image = tf.cond(tf.equal(random_ordering, 3),
                        lambda: distort_way_3(image), lambda: image)
        image = tf.cond(tf.equal(random_ordering, 4),
                        lambda: distort_way_4(image), lambda: image)
        return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_image(image, out_shape=None, data_format='channels_last', to_bgr=True):
    if out_shape is None:
        out_shape = [224, 224]
    with tf.name_scope(name='random_center_crop', values=[image]):
        random_ratio = random.uniform(0.75, 1)
        random_crop_cond = tf.less(tf.random_uniform([], 0, 1.0), 0.5)
        image = tf.cond(random_crop_cond,
                        lambda: tf.image.central_crop(image, random_ratio), lambda: image)

    image = tf.image.resize_images(image, size=out_shape, method=tf.image.ResizeMethod.BILINEAR)

    with tf.name_scope(name='random_flip_left_right', values=[image]):
        mirror_cond = tf.less(tf.random_uniform([], 0, 1.0), 0.5)
        image = tf.cond(mirror_cond, lambda: image, lambda: tf.image.flip_left_right(image))

    with tf.name_scope(name='random_flip_up_down', values=[image]):
        mirror_cond = tf.less(tf.random_uniform([], 0, 1.0), 0.5)
        image = tf.cond(mirror_cond, lambda: image, lambda: tf.image.flip_up_down(image))

    with tf.name_scope(name='random_rot90', values=[image]):
        mirror_cond = tf.less(tf.random_uniform([], 0, 1.0), 0.5)
        image = tf.cond(mirror_cond, lambda: image, lambda: tf.image.rot90(image))

    with tf.name_scope(name='random_distort_color', values=[image]):
        distort_cond = tf.less(tf.random_uniform([], 0, 1.0), .5)
        image = tf.cond(distort_cond, lambda: tf.image.random_brightness(image, max_delta=0.2),
                        lambda: image)
        distort_cond = tf.less(tf.random_uniform([], 0, 1.0), .5)
        image = tf.cond(distort_cond, lambda: tf.image.random_contrast(image, lower=0.8, upper=1.2),
                        lambda: image)
        distort_cond = tf.less(tf.random_uniform([], 0, 1.0), .5)
        image = tf.cond(distort_cond, lambda: tf.image.random_hue(image, max_delta=0.2),
                        lambda: image)
        distort_cond = tf.less(tf.random_uniform([], 0, 1.0), .5)
        image = tf.cond(distort_cond,
                        lambda: tf.image.random_saturation(image, lower=0.8, upper=1.2),
                        lambda: image)
    image = tf.clip_by_value(image, 0, 1.0)
    r, g, b = tf.unstack(image, axis=-1)
    image = tf.cond(tf.constant(to_bgr),
                    lambda: tf.stack([b, g, r], axis=-1), lambda: image, name="rgb_format")
    image = tf.cond(tf.equal(data_format, "channels_last"),
                    lambda: image, lambda: tf.transpose(image, perm=(1, 2, 0)), name="transpose_channel")

    return image


def get_data_from_records_1():
    pass


def get_data_from_records_2():
    pass


def get_data_from_records_3(data_sources, num_samples, batch_size, out_shape, num_classes, label2class,
                            is_train=True, aug=True, data_format="channels_last",
                            num_epochs=None, num_readers=4, num_threads=8):
    keys_to_features = {
        'image/encode': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, 'jpeg'),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
        'label': tf.FixedLenFeature((), tf.int64, default_value=-1)
    }
    items_to_handles = {
        'image': slim.tfexample_decoder.Image('image/encode', 'image/format'),
        'label': slim.tfexample_decoder.Tensor('label'),
        'filename': slim.tfexample_decoder.Tensor('image/filename')
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handles)
    slim_dataset = slim.dataset.Dataset(
        data_sources=data_sources,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=num_samples,
        items_to_descriptions=None,
        num_classes=num_classes,
        labels_to_names=label2class)

    with tf.name_scope('dataset_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            slim_dataset,
            num_readers=num_readers,
            common_queue_capacity=16 * batch_size,
            common_queue_min=8 * batch_size,
            shuffle=is_train,
            num_epochs=num_epochs)

    [image, label, filename] = provider.get(['image', 'label', 'filename'])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if aug:
        image = preprocess_image(image, out_shape, data_format=data_format, to_bgr=True)
    else:
        image = tf.image.resize_images(image, size=out_shape, method=tf.image.ResizeMethod.BILINEAR)
    if data_format == "channels_last":
        image.set_shape(out_shape+[3])
    else:
        image.set_shape([3]+out_shape)

    # batch = tf.train.batch([image, image_draw, filename],
    #                        dynamic_pad=False,
    #                        batch_size=batch_size,
    #                        allow_smaller_final_batch=(not is_training),
    #                        num_threads=num_preprocessing_threads,
    #                        capacity=4 * batch_size)

    batch = tf.train.shuffle_batch([image, label, filename],
                                   batch_size=batch_size,
                                   allow_smaller_final_batch=False,
                                   num_threads=num_threads,
                                   capacity=4 * batch_size,
                                   min_after_dequeue=batch_size)
    batch_queue = slim.prefetch_queue.prefetch_queue(batch, capacity=4, num_threads=num_threads)
    return batch_queue.dequeue()


if __name__ == "__main__":

    dataset_dir = "/media/kun/4DDAE1651159A0A8/dataset/garbage_classify/train_data"
    output_dir = "/home/kun/PycharmProjects"
    out_file = "garbage.tfrecord"
    txt_paths = glob.glob(os.path.join(dataset_dir, "*txt"))
    print(txt_paths.__len__())
    label_map = defaultdict(int)
    for txt_path in txt_paths:
        with open(txt_path, 'r') as f:
            line = f.readline().split(', ')
        label_map[line[1]] += 1
    print(label_map.values())

    # make_records(dataset_dir, output_dir, out_file)

    # dataset = tf.data.TFRecordDataset(os.path.join(output_dir, out_file))
    # dataset = dataset.make_one_shot_iterator()
    # data = dataset.get_next()
    # with tf.Session() as sess:
    #     print(sess.run(data))

    label2class_dict = {
        0: "其他垃圾/一次性快餐盒",
        1: "其他垃圾/污损塑料",
        2: "其他垃圾/烟蒂",
        3: "其他垃圾/牙签",
        4: "其他垃圾/破碎花盆及碟碗",
        5: "其他垃圾/竹筷",
        6: "厨余垃圾/剩饭剩菜",
        7: "厨余垃圾/大骨头",
        8: "厨余垃圾/水果果皮",
        9: "厨余垃圾/水果果肉",
        10: "厨余垃圾/茶叶渣",
        11: "厨余垃圾/菜叶菜根",
        12: "厨余垃圾/蛋壳",
        13: "厨余垃圾/鱼骨",
        14: "可回收物/充电宝",
        15: "可回收物/包",
        16: "可回收物/化妆品瓶",
        17: "可回收物/塑料玩具",
        18: "可回收物/塑料碗盆",
        19: "可回收物/塑料衣架",
        20: "可回收物/快递纸袋",
        21: "可回收物/插头电线",
        22: "可回收物/旧衣服",
        23: "可回收物/易拉罐",
        24: "可回收物/枕头",
        25: "可回收物/毛绒玩具",
        26: "可回收物/洗发水瓶",
        27: "可回收物/玻璃杯",
        28: "可回收物/皮鞋",
        29: "可回收物/砧板",
        30: "可回收物/纸板箱",
        31: "可回收物/调料瓶",
        32: "可回收物/酒瓶",
        33: "可回收物/金属食品罐",
        34: "可回收物/锅",
        35: "可回收物/食用油桶",
        36: "可回收物/饮料瓶",
        37: "有害垃圾/干电池",
        38: "有害垃圾/软膏",
        39: "有害垃圾/过期药物"
    }

    image_tensor, label_tensor, filename_tensor = get_data_from_records_3(
        os.path.join(output_dir, out_file),
        num_samples=14802, batch_size=32, out_shape=[224, 224],
        num_classes=40, label2class=label2class_dict,
        is_train=False, aug=True, data_format="channels_last",
        num_epochs=None, num_readers=4, num_threads=8
    )
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    wait_t = 0
    with tf.Session(config=config) as sess:
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess, coord)
        while True:
            image_val, label_val, filename_val = sess.run(
                [image_tensor, label_tensor, filename_tensor])
            k = 0
            print(filename_val[k], label_val[k], label2class_dict[label_val[k]])
            cv2.imshow("image", image_val[k])
            key = cv2.waitKey(wait_t)
            if key == 27:
                break
            elif key == 32:
                wait_t = 1-wait_t
        coord.request_stop()
        coord.join(thread)