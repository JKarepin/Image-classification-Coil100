import TFDB
import tensorflow as tf
import tensorflow.contrib as tfc

train, test, validation = TFDB.dataset.image.coil100('tmp/coil100')

# values

N = 200
K = 6
L = 10
M = 10
P = 14
R = 12
img_size = 128
num_channels = 3
num_classes = 100
batch_size = 1
print(train)
train_dataset = train.batch(batch_size)
train_iterator = train_dataset.make_initializable_iterator()
train_imgs, train_labels = train_iterator.get_next()
train_labels_one_hot = tf.one_hot(train_labels, num_classes, 1, 0)

test_dataset = test.batch(batch_size)
test_iterator = test_dataset.make_initializable_iterator()
test_imgs, test_labels = test_iterator.get_next()
test_labels_one_hot = tf.one_hot(test_labels, num_classes, 1, 0)

validation_dataset = validation.batch(batch_size)
validation_iterator = validation_dataset.make_initializable_iterator()
validation_imgs, validation_labels = validation_iterator.get_next()
validation_labels_one_hot = tf.one_hot(validation_labels, num_classes, 1, 0)


def weights(output_size, nazwa):
    weight = tf.get_variable(nazwa, output_size, tf.float32, tfc.layers.xavier_initializer())
    return weight


def biases(output_size, nazwa):
    bias = tf.get_variable(nazwa, output_size, tf.float32, tfc.layers.xavier_initializer())
    return bias


def convolutional_layer(input, weigth, bias):
    layer = tf.nn.conv2d(input, weigth, strides=[1, 1, 1, 1], padding='SAME') + bias
    return layer


def convolutional_layer_2(input, weigth, bias):
    layer = tf.nn.conv2d(input, weigth, strides=[1, 2, 2, 1], padding='SAME') + bias
    return layer


def max_pool(input):
    layer = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return layer


def model(img):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        W1 = weights([5, 5, 3, K], "1_w")
        B1 = biases(K, "first_bias")
        W2 = weights([5, 5, K, L], "2_w")
        B2 = biases(L, "sec_bias")
        W3 = weights([4, 4, L, M], "3_w")
        B3 = biases(M, "third_bias")
        W4 = weights([4, 4, M, P], "4_w")
        B4 = biases(P, "fourth_bias")
        W5 = weights([8 * 8 * P, N], "5_w")
        B5 = biases(N, "fifth_bias")
        W6 = weights([N, 100], "6_w")
        B6 = biases(100, "sixth_bias")

        img = tf.cast(img, tf.float32)
        Y1 = tf.nn.relu(convolutional_layer(img, W1, B1))

        # output 64x64  Y1 = 10x128x128x6 W2 = 5,5,6,10
        Y2 = convolutional_layer(Y1, W2, B2)
        Y2 = max_pool(Y2)
        Y2 = tf.nn.relu(Y2)

        # output 32x32  Y2 = 10x64x64x10 W3 = 5,5,10,14
        Y3 = convolutional_layer(Y2, W3, B3)
        Y3 = max_pool(Y3)
        Y3 = tf.nn.relu(Y3)

        # output 16x16  Y3 = 10x32x32x14 W4 = 5,5,14,18
        Y4 = convolutional_layer_2(Y3, W4, B4)
        Y4 = max_pool(Y4)
        Y4 = tf.nn.relu(Y4)

        YY = tf.reshape(Y4, shape=[-1, 8 * 8 * P])
        fc1 = tf.nn.relu(tf.matmul(YY, W5) + B5)

        output = tf.matmul(fc1, W6) + B6
        pred = tf.nn.softmax(output)
        return output, pred


train_output, train_pred = model(train_imgs)
test_output, test_pred = model(test_imgs)
validation_output, validation_pred = model(validation_imgs)

# train

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=train_output, labels=train_labels_one_hot)

loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

correct_prediction = tf.equal(tf.argmax(train_pred, 1), tf.cast(train_labels, tf.int64))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# test

cross_entropy_t = tf.nn.softmax_cross_entropy_with_logits_v2(logits=test_output, labels=test_labels_one_hot)

loss_t = tf.reduce_mean(cross_entropy_t)

correct_prediction_t = tf.equal(tf.argmax(test_pred, 1), tf.cast(test_labels, tf.int64))

accuracy_t = tf.reduce_mean(tf.cast(correct_prediction_t, tf.float32))

# validation

cross_entropy_v = tf.nn.softmax_cross_entropy_with_logits_v2(logits=validation_output, labels=validation_labels_one_hot)

loss_v = tf.reduce_mean(cross_entropy_v)

correct_prediction_v = tf.equal(tf.argmax(validation_pred, 1), tf.cast(validation_labels, tf.int64))

accuracy_v = tf.reduce_mean(tf.cast(correct_prediction_v, tf.float32))

# tensorboard
# dataset api

train_accuracy = tf.summary.scalar('metrics/accuracy', accuracy)
train_loss = tf.summary.scalar('metrics/loss', loss)
stats = tf.summary.merge([train_accuracy, train_loss])

test_accuracy = tf.summary.scalar('metrics/accuracy_t', accuracy_t)
test_loss = tf.summary.scalar('metrics/loss_t', loss_t)
stats_t = tf.summary.merge([test_accuracy, test_loss])

validation_accuracy = tf.summary.scalar('metrics/accuracy_v', accuracy_v)
validation_loss = tf.summary.scalar('metrics/loss_v', loss_v)
stats_v = tf.summary.merge([validation_accuracy, validation_loss])

fwtrain = tf.summary.FileWriter(logdir='./training', graph=tf.get_default_graph())
fwtest = tf.summary.FileWriter(logdir='./testing', graph=tf.get_default_graph())
fwvalidation = tf.summary.FileWriter(logdir='./validation', graph=tf.get_default_graph())

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    i = 0
    j = 0
    for epoch in range(80):
        sess.run(train_iterator.initializer)
        while True:
            try:
                one_hot, out, pred = sess.run([train_labels_one_hot, train_output,train_pred])
                print(one_hot,"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", out,"###########",pred)
                #print("y", epoch)
                #print("$###################", train_output, "##############", train_pred)
                _, o_stats = sess.run([optimizer, stats])
                fwtrain.add_summary(o_stats, i)
                i += 1
            except tf.errors.OutOfRangeError:
                break
        sess.run(validation_iterator.initializer)
        while True:
            try:
                validation_stats = sess.run(stats_v)
                fwvalidation.add_summary(validation_stats, j)
                j += 1

            except tf.errors.OutOfRangeError:
                break
    sess.run(test_iterator.initializer)
    while True:
        try:
            test_stats = sess.run(stats_t)
            fwtest.add_summary(test_stats, i)
            i += 1

        except tf.errors.OutOfRangeError:
            break