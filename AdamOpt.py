import numpy as np
import tensorflow as tf
from itertools import product

class AdamOpt:

    """
    Example of use:

    optimizer = AdamOpt.AdamOpt(variables, learning_rate=self.lr)
    self.train = optimizer.compute_gradients(self.loss, gate=0)
    gvs = optimizer.return_gradients()
    self.g = gvs[0][0]
    self.v = gvs[0][1]
    """

    def __init__(self, variables, algorithm = 'adam', learning_rate = 0.001):

        if algorithm == 'adam':
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-08
            self.t = 0
            self.variables = variables
            self.learning_rate = learning_rate

            self.m = {}
            self.v = {}
            self.delta_grads = {}
            for var in self.variables:
                self.m[var.op.name]  = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
                self.v[var.op.name]  = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
                self.delta_grads[var.op.name]  = tf.Variable(tf.zeros(var.get_shape()), trainable=False)

        elif algorithm == 'rmsprop':
            self.t = 0
            self.alpha = 0.9
            self.epsilon = 1e-10
            self.variables = variables
            self.learning_rate = learning_rate
            self.v = {}
            self.delta_grads = {}
            for var in self.variables:
                self.v[var.op.name]  = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
                self.delta_grads[var.op.name]  = tf.Variable(tf.zeros(var.get_shape()), trainable=False)


        self.grad_descent = tf.train.GradientDescentOptimizer(learning_rate = 1.0)


    def reset_params(self):

        self.t = 0
        reset_op = []
        for var in self.variables:
            reset_op.append(tf.assign(self.v[var.op.name], tf.zeros(var.get_shape())))
            reset_op.append(tf.assign(self.delta_grads[var.op.name], tf.zeros(var.get_shape())))
            if algorithm == 'adam':
                reset_op.append(tf.assign(self.m[var.op.name], tf.zeros(var.get_shape())))

        return tf.group(*reset_op)

    def reset_delta_grads(self):

        reset_op = []
        for var in self.variables:
            reset_op.append(tf.assign(self.delta_grads[var.op.name], tf.zeros(var.get_shape())))

        return tf.group(*reset_op)


    def optimize(self, loss):

        grads_and_vars = self.compute_gradients(loss)
        train_op = self.apply_gradients(grads_and_vars)

        return train_op

    def compute_gradients_rmsprop(self, loss):

        self.gradients = self.grad_descent.compute_gradients(loss, var_list = self.variables)
        self.t += 1
        update_var_op = []

        for (grads, _), var in zip(self.gradients, self.variables):
            update_var_op.append(tf.assign_add(self.delta_grads[var.op.name], grads))

        return tf.group(*update_var_op)

    def update_weights_rmsprop(self, lr_multiplier = 1.):

        update_weights_op = []
        for var in self.variables:
            grads = self.delta_grads[var.op.name] / self.t
            new_v = self.alpha*self.v[var.op.name] + (1-self.alpha)*grads*grads
            delta_grad = - lr_multiplier*self.learning_rate*grads/(tf.sqrt(new_v) + self.epsilon)
            update_weights_op.append(tf.assign_add(var, delta_grad))
            update_weights_op.append(tf.assign(self.v[var.op.name],new_v))
        with tf.control_dependencies(update_weights_op):
            update_weights_op.append(self.reset_delta_grads())
            self.t = 0

        return tf.group(*update_weights_op)


    def compute_gradients(self, loss, apply_gradients = False, lr_multiplier = 1.):

        self.gradients = self.grad_descent.compute_gradients(loss, var_list = self.variables)

        self.t += 1
        lr = self.learning_rate*np.sqrt(1-self.beta2**self.t)/(1-self.beta1**self.t)
        update_var_op = []

        #grads_and_vars = []
        for (grads, _), var in zip(self.gradients, self.variables):
            #new_m = self.beta1*self.m[var.op.name] + (1-self.beta1)*grads
            #new_v = self.beta2*self.v[var.op.name] + (1-self.beta2)*grads*grads

            #delta_grad = - lr*new_m/(tf.sqrt(new_v) + self.epsilon)
            #delta_grad = tf.clip_by_norm(delta_grad, 1)

            new_m = 1.*grads
            new_v = 0.99*self.v[var.op.name] + 0.01*grads*grads

            delta_grad = - lr_multiplier*lr*new_m/(tf.sqrt(new_v) + self.epsilon)
            delta_grad = tf.clip_by_norm(delta_grad, 1)

            update_var_op.append(tf.assign(self.m[var.op.name], new_m))
            update_var_op.append(tf.assign(self.v[var.op.name], new_v))

            update_var_op.append(tf.assign_add(self.delta_grads[var.op.name], delta_grad))
            if apply_gradients:
                update_var_op.append(tf.assign_add(var, delta_grad))

        return tf.group(*update_var_op)

    def update_weights(self):

        update_weights_op = []
        for (grads, _), var in zip(self.gradients, self.variables):
            update_weights_op.append(tf.assign_add(var, self.delta_grads[var.op.name]))
        with tf.control_dependencies(update_weights_op):
            update_weights_op.append(self.reset_delta_grads())

        return tf.group(*update_weights_op)

    def return_delta_grads(self):
        return self.delta_grads

    def return_means(self):
        return self.m

    def return_grads_and_vars(self):
        return self.gradients
