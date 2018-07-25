import tensorflow as tf
import matplotlib.pyplot as plt

tf.reset_default_graph()

input_data = tf.placeholder(dtype=tf.float32, shape=None)
output_data = tf.placeholder(dtype=tf.float32, shape=None)

slope = tf.Variable(0.5, dtype=tf.float32)
intercept = tf.Variable(0.1, dtype=tf.float32)

model_operation = input_data * slope + intercept

error = model_operation - output_data
squared_error = tf.square(error)
loss = tf.reduce_mean(squared_error)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

start_values = [0, 1, 2, 3, 4]
end_values = [1, 3, 5, 7, 9]

with tf.Session() as sess:
    sess.run(init)
    trainCount = 200000
    previousProgress = -1
    for i in range(trainCount):
        sess.run(train, feed_dict={input_data: start_values, output_data: end_values})
        if i % 100 == 0:
            if int(i / trainCount * 100) > previousProgress:
                previousProgress = int(i / trainCount * 100)
                print("Progress: " + str(previousProgress) + "%")
            plt.plot(start_values, sess.run(model_operation, feed_dict={input_data: start_values}))
    print(sess.run([slope, intercept]))
    print("Loss: " + str(sess.run(loss, feed_dict={input_data: start_values, output_data: end_values})))
    plt.plot(start_values, end_values, 'ro', 'Training Data')
    plt.plot(start_values, sess.run(model_operation, feed_dict={input_data: start_values}))
    plt.show()
