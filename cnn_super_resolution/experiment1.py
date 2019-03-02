# copyright (c) 2019 K Sreram, All rights reserved.
import cv2
import os

import tensorflow as tf

from prepare_dataset.sr_image_ds_manager import ImageDSManage

from datetime import datetime

from prepare_dataset.split_dataset import SplitData

from model_saver_manager import model_saver


class Model1:

    def __init__(self):
        self.__input_data = None
        self.__expected_output_data = None

        self.__parameters = None

        self.__model = None

        self.__output_img_var = None

        self.__loss1 = None

        self.__loss2 = None

        self.__output_img = None

    def create_place_holders(self):
        self.__input_data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None])
        self.__expected_output_data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None])
        return self.__input_data, self.__expected_output_data

    def create_parameters(self):
        self.__output_img_var = tf.Variable(initial_value=tf.truncated_normal(shape=[400, 400, 3]))
        return self.__output_img_var

    def create_model(self):

        if self.__output_img_var is not None:
            output_img = self.__output_img_var

            output_img = tf.square(output_img)

            self.__output_img = tf.divide(output_img, tf.reduce_max(output_img))

            return output_img

    def create_loss_1(self):
        temp1 = tf.subtract(self.__input_data, self.__expected_output_data)
        temp2 = tf.subtract(self.__input_data, self.__output_img)
        temp1 = tf.square(temp1)
        temp2 = tf.square(temp2)

        loss = tf.subtract(temp1, temp2)
        loss = tf.square(loss)

        self.__loss1 = tf.reduce_mean(loss)
        return self.__loss1

    def create_loss_2(self):
        if self.__loss1 is not None:
            loss2 = tf.square(tf.subtract(self.__expected_output_data, self.__output_img))
            loss2 = tf.reduce_mean(loss2)
            self.__loss2 = tf.add(self.__loss1, loss2)
            return self.__loss2


class Model2:

    def __init__(self):
        self.__input_data = None
        self.__expected_output_data = None

        self.__filters = []

        self.__biases = []

        self.__model = None

        self.__loss1 = None

        self.__loss2 = None

        self.__loss3 = None
        self.__loss4 = None

        self.__padding = "SAME"

        self.__strides = [1, 1, 1, 1]

    def create_place_holders(self):
        self.__input_data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None])
        self.__expected_output_data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None])
        return self.__input_data, self.__expected_output_data

    def create_parameters(self):

        self.__filters.append(tf.Variable(initial_value=tf.truncated_normal(shape=[10, 10, 3, 80])))
        self.__filters.append(tf.Variable(initial_value=tf.truncated_normal(shape=[2, 2, 80, 40])))
        self.__filters.append(tf.Variable(initial_value=tf.truncated_normal(shape=[10, 10, 40, 3])))

        self.__biases.append(tf.Variable(initial_value=tf.truncated_normal(shape=[80])))
        self.__biases.append(tf.Variable(initial_value=tf.truncated_normal(shape=[40])))
        self.__biases.append(tf.Variable(initial_value=tf.truncated_normal(shape=[3])))

        return self.__filters, self.__biases

    def create_model(self):

        if len(self.__filters) > 0 and len(self.__biases) > 0:
            cur_layer = tf.nn.conv2d(input=self.__input_data, filter=self.__filters[0],
                                     strides=self.__strides, padding=self.__padding)
            cur_layer = tf.add(cur_layer, self.__biases[0])
            cur_layer = tf.nn.relu(cur_layer)

            cur_layer = tf.nn.conv2d(input=cur_layer, filter=self.__filters[1],
                                     strides=self.__strides, padding=self.__padding)
            cur_layer = tf.add(cur_layer, self.__biases[1])
            cur_layer = tf.nn.relu(cur_layer)

            cur_layer = tf.nn.conv2d(input=cur_layer, filter=self.__filters[2],
                                     strides=self.__strides, padding=self.__padding)
            cur_layer = tf.add(cur_layer, self.__biases[2])
            cur_layer = tf.nn.relu(cur_layer)

            cur_layer = tf.divide(cur_layer, tf.reduce_max(cur_layer))

            self.__model = cur_layer

            return cur_layer

    def create_loss_1(self):
        temp1 = tf.subtract(self.__input_data, self.__expected_output_data)
        temp2 = tf.subtract(self.__input_data, self.__model)
        temp1 = tf.square(temp1)
        temp2 = tf.square(temp2)

        loss = tf.subtract(temp1, temp2)
        loss = tf.square(loss)

        self.__loss1 = tf.reduce_mean(loss)
        return self.__loss1

    def create_loss_2(self):
        # forces the model to not learn the identity.
        if self.__loss1 is None:
            self.create_loss_1()

        loss2 = tf.square(tf.subtract(self.__expected_output_data, self.__model))
        loss2 = tf.reduce_mean(loss2)
        self.__loss2 = tf.add(self.__loss1, loss2)
        return self.__loss2

    def create_loss_3(self):
        loss3 = tf.square(tf.subtract(self.__expected_output_data, self.__model))
        loss3 = tf.reduce_mean(loss3)
        self.__loss3 = loss3
        return self.__loss3

    def create_loss_4(self):
        if self.__loss3 is None:
            self.create_loss_3()

        loss4 = tf.abs(tf.reduce_mean(self.__filters[0]))
        loss4 = tf.add(loss4, tf.multiply(-1.0, tf.reduce_mean(tf.square(self.__filters[0]))))
        loss4 = tf.add(self.__loss3, loss4)
        self.__loss4 = loss4
        return self.__loss4


def trainer(loss_num, batch_down_sampled, batch_original, name="1"):
    img_manager = ImageDSManage(["/media/sreramk/storage-main/elementary_frame/dataset/DIV2K_train_HR/"],
                                image_buffer_limit=50, buffer_priority=50)

    # input_data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None])
    # expected_output_data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None])
    """
    model1 = Model1()

    input_data, expected_output_data = model1.create_place_holders()

    output_img_var = model1.create_parameters()

    output_img = model1.create_model()

    loss = model1.create_loss_1()

    loss2 = model1.create_loss_2()

    adam = tf.train.AdamOptimizer().minimize(loss, var_list=[output_img_var])
    """
    model2 = Model2()

    input_data, expected_output_data = model2.create_place_holders()

    filters, bisases = model2.create_parameters()

    parameters = list(filters)
    parameters.extend(bisases)

    modelsave = model_saver.ModelSaver('exp1_',parameters,
                    save_file_path="/media/sreramk/storage-main/elementary_frame/model_checkpoints")
    query = modelsave.query_checkpoint_info(check_point_id_range=(0, 100))
    print (query)
    model = model2.create_model()

    if loss_num == 1:
        loss = model2.create_loss_1()
    elif loss_num == 2:
        loss = model2.create_loss_2()
    elif loss_num == 3:
        loss = model2.create_loss_3()
    elif loss_num == 4:
        loss = model2.create_loss_4()

    prams = list(filters)

    prams.extend(bisases)

    adam = tf.train.AdamOptimizer().minimize(loss, var_list=prams)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        init = tf.initialize_all_variables()

        sess.run(init)

        # sess.graph.finalize()

        cv2.imshow("_original" + name, batch_original[0])
        cv2.imshow("_down_sampled" + name, batch_down_sampled[0])
        cv2.waitKey(0)
        # cv2.destroyAllWindows()
        first_time = True
        for i in range(12020):

            min_x, min_y, batch_down_sampled, batch_original = img_manager.get_batch(batch_size=6,
                                                                                     down_sample_factor=4,
                                                                                     min_x_f=70, min_y_f=70)

            lresult = sess.run(fetches=[adam, loss, model],
                               feed_dict={input_data: batch_down_sampled,
                                          expected_output_data: batch_original})

            __, cur_loss, cur_output_img = lresult

            if first_time:
                print(sess.run(fetches=parameters))

            modelsave.checkpoint_model(float(cur_loss), sess=sess, skip_type=modelsave.TimeIterSkipManager.ST_ITER_SKIP,
                                       skip_duration=50)
            if first_time:
                print("#####################################")
                print(print(sess.run(fetches=parameters)))
            first_time = False

            if i % 100 == 0:
                # print(lresult)
                print("loss = " + str(cur_loss))
                print("i = " + str(i))

        min_x, min_y, batch_down_sampled, batch_original = img_manager.get_batch(batch_size=6,
                                                                                 down_sample_factor=4,
                                                                                 min_x_f=500, min_y_f=500)

        cur_output_img = sess.run(fetches=[model], feed_dict={input_data: batch_down_sampled})

        cv2.imshow("original" + name, batch_original[0])
        cv2.imshow("down_sampled" + name, batch_down_sampled[0])
        cv2.imshow("computed" + name, cur_output_img[0][0])

        cv2.waitKey(0)

        # cv2.destroyAllWindows()


if __name__ == "__main__":
    img_manager = ImageDSManage(["/media/sreramk/storage-main/elementary_frame/dataset/DIV2K_train_HR/"],
                                image_buffer_limit=10, buffer_priority=1)
    min_x, min_y, batch_down_sampled, batch_original = img_manager.get_batch(batch_size=6,
                                                                             down_sample_factor=4,
                                                                             min_x_f=70, min_y_f=70)
    # trainer(1, batch_down_sampled, batch_original)
    trainer(2, batch_down_sampled, batch_original, "2")
    # trainer(3, batch_down_sampled, batch_original, "3")
    cv2.destroyAllWindows()
