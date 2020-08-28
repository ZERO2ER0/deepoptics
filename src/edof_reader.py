import numpy as np
import tensorflow as tf
import os
import pdb

from glob import glob


def get_edof_training_queue(target_dir, patch_size, batch_size, num_depths=4, color=False,
                            num_threads=4, loop=True, filetype='jpg'):
    if filetype == 'jpg':
        file_list = tf.matching_files(os.path.join(target_dir, '*.jpg'))
    elif filetype == 'png':
        file_list = tf.matching_files(os.path.join(target_dir, '*.png'))

    filename_queue = tf.train.string_input_producer(file_list,
                                                    num_epochs=None if loop else 1,
                                                    shuffle=True if loop else False)

    image_reader = tf.WholeFileReader()

    _, image_file = image_reader.read(filename_queue)
    if filetype == 'jpg':
        if color:
            print("Using color images")
            image = tf.image.decode_jpeg(image_file,
                                         channels=0)
        else:
            print("Using black and white images")
            image = tf.image.decode_jpeg(image_file,
                                         channels=1)
    elif filetype == 'png':
        if color:
            print("Using color images")
            image = tf.image.decode_png(image_file,
                                        channels=0)
        else:
            print("Using black and white images")
            image = tf.image.decode_png(image_file,
                                        channels=1)

    image = tf.cast(image, tf.float32)  # Shape [height, width, 1]
    image = tf.expand_dims(image, 0)
    image /= 255.

    # Get the ratio of the patch size to the smallest side of the image
    img_height_width = tf.cast(tf.shape(image)[1:3], tf.float32)

    size_ratio = patch_size / tf.reduce_min(img_height_width)

    # Extract a glimpse from the image
    offset_center = tf.random_uniform([1, 2], minval=0.0 + size_ratio / 2, maxval=1.0 - size_ratio / 2,
                                      dtype=tf.float32)
    offset_center = offset_center * img_height_width
    pdb.set_trace()
    image = tf.image.extract_glimpse(image, size=[patch_size, patch_size], offsets=offset_center, centered=False,
                                     normalized=False)
    image = tf.squeeze(image, 0)

    all_depths = tf.convert_to_tensor([1 / 2, 1 / 1.5, 1 / 1, 1 / 0.5, 1000], tf.float32)

    depth_bins = []
    for i in range(num_depths):
        depth_idx = tf.multinomial(tf.log([5 * [1 / 5]]), num_samples=1)
        depth_bins.append(all_depths[tf.cast(depth_idx[0][0], tf.int32)])

    test_depth = np.concatenate(
        [np.ones((patch_size // len(depth_bins), patch_size)) * i for i in range(len(depth_bins))], axis=0)[:, :, None]

    if color:
        patch_dims = [patch_size, patch_size, 3]
    else:
        patch_dims = [patch_size, patch_size, 1]

    image_batch, depth_batch = tf.train.batch([image, test_depth],
                                              shapes=[patch_dims, [patch_size, patch_size, 1]],
                                              batch_size=batch_size,
                                              num_threads=num_threads,
                                              capacity=4 * batch_size)
    tf.summary.image("input_img", image_batch)
    tf.summary.scalar("input_img_max", tf.reduce_max(image_batch))
    tf.summary.scalar("input_img_min", tf.reduce_min(image_batch))
    tf.summary.histogram('depth', depth_bins)
    tf.summary.image('depth', tf.cast(depth_batch, tf.float32))

    return image_batch, depth_batch, depth_bins

def get_nyu_training_queue(target_dir, patch_size, batch_size, num_depths=4, color=False,
                            num_threads=4, loop=True, filetype='png'):

    image_dir = os.path.join(target_dir, 'rgb')
    depth_dir = os.path.join(target_dir, 'depth_bin')
    depth_idx_dir = os.path.join(target_dir, 'depth_idx')
    imagesName = os.listdir(image_dir)
    image_list=[]
    depth_list=[]
    depth_idx_list=[]
    for imageName in imagesName:
        image_path = os.path.join(image_dir, imageName)
        depth_path = os.path.join(depth_dir, imageName)
        depth_idx_path = os.path.join(depth_idx_dir, imageName)
        image_list.append(image_path)
        depth_list.append(depth_path)
        depth_idx_list.append(depth_idx_path)

    image_list = tf.convert_to_tensor(image_list, dtype=tf.string)
    depth_list = tf.convert_to_tensor(depth_list, dtype=tf.string)
    depth_idx_list = tf.convert_to_tensor(depth_idx_list, dtype=tf.string)

    image_queue, depth_queue, depth_idx_queue = tf.train.slice_input_producer(
                                                    [image_list, depth_list, depth_idx_list],
                                                    num_epochs=None if loop else 1,
                                                    shuffle=True if loop else False)
   
    # image_reader = tf.WholeFileReader()
    depths_min = 0.7132995
    depths_max = 9.99547
    disc_depths_bin =[0]*12
    depths_bin = np.linspace(depths_min, depths_max, 13)
    for i in range(len(depths_bin)-1):
        disc_depths_bin[i] = (depths_bin[i] + depths_bin[i+1])/2.

    image_file = tf.read_file(image_queue)
    depth_file = tf.read_file(depth_queue)
    depth_idx_file = tf.read_file(depth_idx_queue)

    if filetype == 'jpg':
        if color:
            print("Using color images")
            image = tf.image.decode_jpeg(image_file,
                                         channels=0)
        else:
            print("Using black and white images")
            image = tf.image.decode_jpeg(image_file,
                                         channels=1)
        depth = tf.image.decode_jpeg(depth_file, channels =1)
        depth_idx = tf.image.decode_jpeg(depth_idx_file, channels =1)
    elif filetype == 'png':
        if color:
            print("Using color images")
            image = tf.image.decode_png(image_file,
                                        channels=0)
        else:
            print("Using black and white images")
            image = tf.image.decode_png(image_file,
                                        channels=1)
        depth = tf.image.decode_png(depth_file, channels =1, dtype=tf.uint16)
        depth_idx = tf.image.decode_png(depth_idx_file, channels =1)

    image = tf.cast(image, tf.float32)  # Shape [height, width, 1]
    depth = tf.cast(depth, tf.float32)
    depth_idx = tf.cast(depth_idx, tf.float32)
    # image = tf.expand_dims(image, 0)
    image /= 255.
    # depth /= 1000.
    # depth *=255.
    # Extract a glimpse from the image
    # offset_center = tf.random_uniform([1, 2],  -80, 80,
    #                                   dtype=tf.float32)
    # offset_center = offset_center*[0,1]

    # image = tf.image.extract_glimpse(tf.expand_dims(image, 0), 
    #                                 size=[patch_size, patch_size], offsets=offset_center, centered=True,
    #                                 normalized=False)

    # depth_idx = tf.image.extract_glimpse(tf.expand_dims(depth_idx, 0), 
    #                                     size=[patch_size, patch_size], offsets=offset_center, centered=True,
    #                                     normalized=False)
    # image = tf.squeeze(image, 0)
    # depth_idx = tf.squeeze(depth_idx, 0)


    if color:
        image_dims = [480, 640, 3]
    else:
        image_dims = [480, 640, 1]
    depth_dims = [480, 640, 1]

    image_batch, depth_batch, depth_idx_batch = tf.train.batch([image, depth, depth_idx],
                                              shapes=[image_dims, depth_dims, depth_dims],
                                              batch_size=batch_size,
                                              num_threads=num_threads,
                                              capacity=4 * batch_size)

    # # Get the ratio of the patch size to the smallest side of the image
    # img_height_width = tf.cast(tf.shape(image_batch)[1:3], tf.float32)
    # size_ratio = patch_size / tf.reduce_min(img_height_width)

    # Extract a glimpse from the image
    offset_center = tf.random_uniform([1, 2],  -80, 80,
                                      dtype=tf.float32)
    offset_center = offset_center*[0,1]

    image_batch = tf.image.extract_glimpse(image_batch, 
                                    size=[patch_size, patch_size], offsets=offset_center, centered=True,
                                    normalized=False)

    depth_idx_batch = tf.image.extract_glimpse(depth_idx_batch, 
                                        size=[patch_size, patch_size], offsets=offset_center, centered=True,
                                        normalized=False)
    tf.summary.image("input_img", image_batch)
    tf.summary.scalar("input_img_max", tf.reduce_max(image_batch))
    tf.summary.scalar("input_img_min", tf.reduce_min(image_batch))
    # tf.summary.histogram('depth', depth_bins)
    tf.summary.image('depth', tf.cast(depth_batch, tf.float32))
    tf.summary.scalar("depth_max", tf.reduce_max(depth_batch))
    tf.summary.scalar("depth_min", tf.reduce_min(depth_batch))

    tf.summary.image('depth_idx', tf.cast(depth_idx_batch, tf.float32))
    tf.summary.scalar("depth_idx_max", tf.reduce_max(depth_idx_batch))
    tf.summary.scalar("depth_idx_min", tf.reduce_min(depth_idx_batch))

    return image_batch, depth_idx_batch, disc_depths_bin

def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
  image = tf.cast(image_decoded, tf.float32)

  image = tf.image.resize_images(image, [224, 224])  # (2)
  return image, filename, label
