import tensorflow as tf
# my_const = tf.constant([1.0, 2.0], name="my_const")
# with tf.Session() as sess:
#     # print(sess.graph.as_graph_def())
#     print(sess.run(my_const))
# # you will see value of my_const stored in the graph’s definition
#
# # Initialize the variables
# W = tf.Variable(tf.zeros([784,10]))
# with tf.Session() as sess:
#     sess.run(W.initializer)
#     print(W)
#     # print(W.eval())
#
#
# my_var = tf.Variable(10)
# with tf.Session() as sess:
#     sess.run(my_var.initializer)
#     # increment by 10
#     sess.run(my_var.assign_add(10)) # >> 20
#     # decrement by 2
#     sess.run(my_var.assign_sub(2)) # >> 18
#     print(my_var.eval())

dict_temp = [1, 2, 3]
with tf.device('/cpu:0'), tf.name_scope("embedding"):
    W = tf.Variable(
        dict_temp,name="W")

with tf.Session() as sess:
    sess.run(W.initializer)
    print(W.eval())

a = [[[1, 2], [3, 4]]]

print(list(a))