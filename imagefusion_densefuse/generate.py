# Use a trained DenseFuse Net to generate fused images

import tensorflow as tf
import numpy as np
from datetime import datetime

from fusion_l1norm import L1_norm
from densefuse_net import DenseFuseNet
from utils import get_images, save_images, get_train_images, get_test_image_rgb


def generate(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, index, IS_VIDEO, IS_RGB, type='addition', output_path=None):

	if IS_VIDEO:
		print('video_addition')
		_handler_video(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, output_path=output_path)
	else:
		if IS_RGB:
			print('RGB - addition')
			_handler_rgb(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, index,
			         output_path=output_path)

			print('RGB - l1')
			_handler_rgb_l1(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, index,
			             output_path=output_path)
		else:
			if type == 'addition':
				print('addition')
				_handler(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, index, output_path=output_path)
			elif type == 'l1':
				print('l1')
				_handler_l1(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, index, output_path=output_path)


def _handler(ir_path, vis_path, model_path, model_pre_path, ssim_weight, index, output_path=None):
	ir_img = get_train_images(ir_path, flag=False)
	vis_img = get_train_images(vis_path, flag=False)
	# ir_img = get_train_images_rgb(ir_path, flag=False)
	# vis_img = get_train_images_rgb(vis_path, flag=False)
	dimension = ir_img.shape

	ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
	vis_img = vis_img.reshape([1, dimension[0], dimension[1], dimension[2]])

	ir_img = np.transpose(ir_img, (0, 2, 1, 3))
	vis_img = np.transpose(vis_img, (0, 2, 1, 3))

	print('img shape final:', ir_img.shape)

	with tf.Graph().as_default(), tf.Session() as sess:
		infrared_field = tf.placeholder(
			tf.float32, shape=ir_img.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=ir_img.shape, name='style')

		dfn = DenseFuseNet(model_pre_path)

		output_image = dfn.transform_addition(infrared_field, visible_field)
		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		output = sess.run(output_image, feed_dict={infrared_field: ir_img, visible_field: vis_img})

		save_images(ir_path, output, output_path,
		            prefix='fused' + str(index), suffix='_densefuse_addition_'+str(ssim_weight))


def _handler_l1(ir_path, vis_path, model_path, model_pre_path, ssim_weight, index, output_path=None):
	ir_img = get_train_images(ir_path, flag=False)
	vis_img = get_train_images(vis_path, flag=False)
	dimension = ir_img.shape

	ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
	vis_img = vis_img.reshape([1, dimension[0], dimension[1], dimension[2]])

	ir_img = np.transpose(ir_img, (0, 2, 1, 3))
	vis_img = np.transpose(vis_img, (0, 2, 1, 3))

	print('img shape final:', ir_img.shape)

	with tf.Graph().as_default(), tf.Session() as sess:

		# build the dataflow graph
		infrared_field = tf.placeholder(
			tf.float32, shape=ir_img.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=ir_img.shape, name='style')

		dfn = DenseFuseNet(model_pre_path)

		enc_ir = dfn.transform_encoder(infrared_field)
		enc_vis = dfn.transform_encoder(visible_field)

		target = tf.placeholder(
		    tf.float32, shape=enc_ir.shape, name='target')

		output_image = dfn.transform_decoder(target)

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		enc_ir_temp, enc_vis_temp = sess.run([enc_ir, enc_vis], feed_dict={infrared_field: ir_img, visible_field: vis_img})
		feature = L1_norm(enc_ir_temp, enc_vis_temp)

		output = sess.run(output_image, feed_dict={target: feature})
		save_images(ir_path, output, output_path,
		            prefix='fused' + str(index), suffix='_densefuse_l1norm_'+str(ssim_weight))


def _handler_video(ir_path, vis_path, model_path, model_pre_path, ssim_weight, output_path=None):
	infrared = ir_path[0]
	img = get_train_images(infrared, flag=False)
	img = img.reshape([1, img.shape[0], img.shape[1], img.shape[2]])
	img = np.transpose(img, (0, 2, 1, 3))
	print('img shape final:', img.shape)
	num_imgs = len(ir_path)

	with tf.Graph().as_default(), tf.Session() as sess:
		# build the dataflow graph
		infrared_field = tf.placeholder(
			tf.float32, shape=img.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=img.shape, name='style')

		dfn = DenseFuseNet(model_pre_path)

		output_image = dfn.transform_addition(infrared_field, visible_field)

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		##################GET IMAGES###################################################################################
		start_time = datetime.now()
		for i in range(num_imgs):
			print('image number:', i)
			infrared = ir_path[i]
			visible = vis_path[i]

			ir_img = get_train_images(infrared, flag=False)
			vis_img = get_train_images(visible, flag=False)
			dimension = ir_img.shape

			ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
			vis_img = vis_img.reshape([1, dimension[0], dimension[1], dimension[2]])

			ir_img = np.transpose(ir_img, (0, 2, 1, 3))
			vis_img = np.transpose(vis_img, (0, 2, 1, 3))

			################FEED########################################
			output = sess.run(output_image, feed_dict={infrared_field: ir_img, visible_field: vis_img})
			save_images(infrared, output, output_path,
			            prefix='fused' + str(i), suffix='_addition_' + str(ssim_weight))
			######################################################################################################
		elapsed_time = datetime.now() - start_time
		print('Dense block video==> elapsed time: %s' % (elapsed_time))


def _handler_rgb(ir_path, vis_path, model_path, model_pre_path, ssim_weight, index, output_path=None):
	# ir_img = get_train_images(ir_path, flag=False)
	# vis_img = get_train_images(vis_path, flag=False)
	ir_img = get_test_image_rgb(ir_path, flag=False)
	vis_img = get_test_image_rgb(vis_path, flag=False)
	dimension = ir_img.shape

	ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
	vis_img = vis_img.reshape([1, dimension[0], dimension[1], dimension[2]])

	#ir_img = np.transpose(ir_img, (0, 2, 1, 3))
	#vis_img = np.transpose(vis_img, (0, 2, 1, 3))

	ir_img1 = ir_img[:, :, :, 0]
	ir_img1 = ir_img1.reshape([1, dimension[0], dimension[1], 1])
	ir_img2 = ir_img[:, :, :, 1]
	ir_img2 = ir_img2.reshape([1, dimension[0], dimension[1], 1])
	ir_img3 = ir_img[:, :, :, 2]
	ir_img3 = ir_img3.reshape([1, dimension[0], dimension[1], 1])

	vis_img1 = vis_img[:, :, :, 0]
	vis_img1 = vis_img1.reshape([1, dimension[0], dimension[1], 1])
	vis_img2 = vis_img[:, :, :, 1]
	vis_img2 = vis_img2.reshape([1, dimension[0], dimension[1], 1])
	vis_img3 = vis_img[:, :, :, 2]
	vis_img3 = vis_img3.reshape([1, dimension[0], dimension[1], 1])

	print('img shape final:', ir_img1.shape)

	with tf.Graph().as_default(), tf.Session() as sess:
		infrared_field = tf.placeholder(
			tf.float32, shape=ir_img1.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=ir_img1.shape, name='style')

		dfn = DenseFuseNet(model_pre_path)

		output_image = dfn.transform_addition(infrared_field, visible_field)
		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		output1 = sess.run(output_image, feed_dict={infrared_field: ir_img1, visible_field: vis_img1})
		output2 = sess.run(output_image, feed_dict={infrared_field: ir_img2, visible_field: vis_img2})
		output3 = sess.run(output_image, feed_dict={infrared_field: ir_img3, visible_field: vis_img3})

		output1 = output1.reshape([1, dimension[0], dimension[1]])
		output2 = output2.reshape([1, dimension[0], dimension[1]])
		output3 = output3.reshape([1, dimension[0], dimension[1]])

		output = np.stack((output1, output2, output3), axis=-1)
		#output = np.transpose(output, (0, 2, 1, 3))
		save_images(ir_path, output, output_path,
		            prefix='fused' + str(index), suffix='_densefuse_addition_'+str(ssim_weight))


def _handler_rgb_l1(ir_path, vis_path, model_path, model_pre_path, ssim_weight, index, output_path=None):
	# ir_img = get_train_images(ir_path, flag=False)
	# vis_img = get_train_images(vis_path, flag=False)
	ir_img = get_test_image_rgb(ir_path, flag=False)
	vis_img = get_test_image_rgb(vis_path, flag=False)
	dimension = ir_img.shape

	ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
	vis_img = vis_img.reshape([1, dimension[0], dimension[1], dimension[2]])

	#ir_img = np.transpose(ir_img, (0, 2, 1, 3))
	#vis_img = np.transpose(vis_img, (0, 2, 1, 3))

	ir_img1 = ir_img[:, :, :, 0]
	ir_img1 = ir_img1.reshape([1, dimension[0], dimension[1], 1])
	ir_img2 = ir_img[:, :, :, 1]
	ir_img2 = ir_img2.reshape([1, dimension[0], dimension[1], 1])
	ir_img3 = ir_img[:, :, :, 2]
	ir_img3 = ir_img3.reshape([1, dimension[0], dimension[1], 1])

	vis_img1 = vis_img[:, :, :, 0]
	vis_img1 = vis_img1.reshape([1, dimension[0], dimension[1], 1])
	vis_img2 = vis_img[:, :, :, 1]
	vis_img2 = vis_img2.reshape([1, dimension[0], dimension[1], 1])
	vis_img3 = vis_img[:, :, :, 2]
	vis_img3 = vis_img3.reshape([1, dimension[0], dimension[1], 1])

	print('img shape final:', ir_img1.shape)

	with tf.Graph().as_default(), tf.Session() as sess:
		infrared_field = tf.placeholder(
			tf.float32, shape=ir_img1.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=ir_img1.shape, name='style')

		dfn = DenseFuseNet(model_pre_path)

		enc_ir = dfn.transform_encoder(infrared_field)
		enc_vis = dfn.transform_encoder(visible_field)

		target = tf.placeholder(
			tf.float32, shape=enc_ir.shape, name='target')

		output_image = dfn.transform_decoder(target)

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		enc_ir_temp, enc_vis_temp = sess.run([enc_ir, enc_vis], feed_dict={infrared_field: ir_img1, visible_field: vis_img1})
		feature = L1_norm(enc_ir_temp, enc_vis_temp)
		output1 = sess.run(output_image, feed_dict={target: feature})

		enc_ir_temp, enc_vis_temp = sess.run([enc_ir, enc_vis], feed_dict={infrared_field: ir_img2, visible_field: vis_img2})
		feature = L1_norm(enc_ir_temp, enc_vis_temp)
		output2 = sess.run(output_image, feed_dict={target: feature})

		enc_ir_temp, enc_vis_temp = sess.run([enc_ir, enc_vis], feed_dict={infrared_field: ir_img3, visible_field: vis_img3})
		feature = L1_norm(enc_ir_temp, enc_vis_temp)
		output3 = sess.run(output_image, feed_dict={target: feature})

		output1 = output1.reshape([1, dimension[0], dimension[1]])
		output2 = output2.reshape([1, dimension[0], dimension[1]])
		output3 = output3.reshape([1, dimension[0], dimension[1]])

		output = np.stack((output1, output2, output3), axis=-1)
		#output = np.transpose(output, (0, 2, 1, 3))
		save_images(ir_path, output, output_path,
		            prefix='fused' + str(index), suffix='_densefuse_l1norm_'+str(ssim_weight))