import tensorflow as tf

a = tf.constant(1)
b = tf.constant(2)

mat1 = tf.ones((4,4))  
mat2 = tf.zeros((4,4))  
mat3 = tf.fill((4,4), 5)
mat4  = tf.random_normal((4,4))

ops = [mat1, mat2, mat3, mat4]
with tf.Session() as sess:
    for o in ops:
        print(sess.run(o))
        print("\n")

