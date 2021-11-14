import tensorflow as tf
import numpy as np

def L1_norm(source_en_a, source_en_b):
    result = []
    narry_a = source_en_a
    narry_b = source_en_b

    dimension = source_en_a.shape

    # caculate L1-norm
    temp_abs_a = tf.abs(narry_a)
    temp_abs_b = tf.abs(narry_b)
    _l1_a = tf.reduce_sum(temp_abs_a,3)
    _l1_b = tf.reduce_sum(temp_abs_b,3)

    _l1_a = tf.reduce_sum(_l1_a, 0)
    _l1_b = tf.reduce_sum(_l1_b, 0)
    l1_a = _l1_a.eval()
    l1_b = _l1_b.eval()

    # caculate the map for source images
    mask_value = l1_a + l1_b

    mask_sign_a = l1_a/mask_value
    mask_sign_b = l1_b/mask_value

    array_MASK_a = mask_sign_a
    array_MASK_b = mask_sign_b

    for i in range(dimension[3]):
        temp_matrix = array_MASK_a*narry_a[0,:,:,i] + array_MASK_b*narry_b[0,:,:,i]
        result.append(temp_matrix)

    result = np.stack(result, axis=-1)

    resule_tf = np.reshape(result, (dimension[0], dimension[1], dimension[2], dimension[3]))

    return resule_tf


