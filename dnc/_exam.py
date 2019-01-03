# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 16:42:06 2019

@author: acer4755g
"""

from utility import utility
import tensorflow as tf

# In[]

# # Exam pairwise_add

with tf.variable_scope("padd") as scope:  
  scope.reuse_variables()
  
  norm_init = tf.random_normal_initializer(mean=0.0, stddev=1.0)
  u = tf.get_variable("u", (10, 1), initializer=norm_init)
  v = tf.get_variable("v", (10, 1), initializer=norm_init)
  
  add = utility.pairwise_add(u, v, False)
  
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    u, v, a = sess.run([u, v, add])
    print(u)
    print(v)
    print(a)
    
# In[]

# # Exam pack_into_tensor

with tf.variable_scope("stack") as scope:
  scope.reuse_variables()
  
  data = tf.reverse_v2(tf.range(10,20,1), axis=[0])
  val_array = tf.TensorArray(tf.int32, 10)
  val_array = val_array.unstack(data)
  cons_array = tf.TensorArray(tf.int32, 10)
  index = tf.constant(0)
  
  def loop_body(index, val_array, cons_array):
    current_value = val_array.read(index)
    
    # notice here we use a array to write into the TensorArray
    cons_array = cons_array.write(index, [current_value, current_value*10])
    index += 1
    return (index, val_array, cons_array)
  
  CONS = tf.while_loop(cond=lambda index, *_: index < 10,
                body=loop_body,
                loop_vars=(index, val_array, cons_array))
  
  CONS_RES_1 = CONS[2].stack()
  CONS_RES_2 = utility.pack_into_tensor(CONS[2], axis=1)
  
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    cons_res_1 = sess.run(CONS_RES_1)
    cons_res_2 = sess.run(CONS_RES_2)
    print(cons_res_1)
    print(cons_res_2)

# In[]

# # Exam unpack_into_tensorarray

with tf.variable_scope("unstack") as scope:
  data1 = tf.range(20,30,1)
  data2 = tf.reverse_v2(tf.multiply(data1,2), [0])
  stackData = tf.stack([data1, data2], axis=1)
  
  unstack_data = utility.unpack_into_tensorarray(stackData, axis=1)
  stack_unstack_data = unstack_data.stack()
  
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    d1, d2, sd = sess.run([data1, data2, stackData])
    print(d1, d2)
    print(sd, sd.shape)
    print(unstack_data)    
    
    sud = sess.run(stack_unstack_data)
    print(sud)
    print(sud.shape)


















