import os
import numpy as np
import time
from math import *
import tensorflow as tf
import scipy.optimize
from numpy import genfromtxt

tf.keras.backend.set_floatx('float64')
tf.get_logger().setLevel('ERROR')

import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nb_f", type=int,help="number of collocation points", default=5000)
parser.add_argument("--k_pneu", type=int,help="number of learning pneu", default = 33)
parser.add_argument("--force", type=float,help="applied force", default=0)
parser.add_argument("--w_c", type=float,help="weight for contact", default=100)
parser.add_argument("--nb_epochs_total", type=int,help="total of ADAM training epochs highest lr", default=0)
parser.add_argument("--vers", type=int,help="version of running")
args = parser.parse_args()

from scipy.spatial import Delaunay

tf.keras.backend.set_floatx('float64')
index_test = [0,2,3,4,5,6,7,8,9,10,11,1,13,25, 37, 49, 12, 20, 23,  27, 30, 34,  40, 45, 50, 53, 55]#
index_train = np.setdiff1d(np.arange(0,60,1), index_test)
X_colloc = np.concatenate((np.load('points_colloc_tire.npy')[index_train]))
contact_line = np.concatenate((np.load('points_contact_tire.npy')[index_train]))

L_normalize = 1
L0 = 100
layers = [int(X_colloc[:, :-2].shape[1])] + 5 * [100] + [2]
E = 1000
nu = 0.3
mu = E / 2 / (1 + nu)
lbda = E * nu / (1 + nu) / (1 - 2 * nu)

unique_z = X_colloc[::2500, 2:]#tf.raw_ops.UniqueV2(x=X_colloc[:, 2:], axis=[0])[0]
radius_array = np.load('radius_tire.npy')
radius_array_final = np.asarray(radius_array)[index_train]

class GADEM_toytire_SVK:
    # Initialize the class
    def __init__(self, X_f, X_contact, X_out, layers):

        self.mu = mu
        self.lbda = lbda

        self.X_f = tf.convert_to_tensor(X_f, dtype='float64')
        self.x_f = tf.convert_to_tensor(X_f[:, 0:1], dtype='float64')
        self.y_f = tf.convert_to_tensor(X_f[:, 1:2], dtype='float64')
        self.z_f = tf.convert_to_tensor(X_f[:, 2:-2], dtype='float64')

        self.X_out = tf.convert_to_tensor(X_out, dtype='float64')
        self.X_contact = tf.convert_to_tensor(X_contact, dtype='float64')
        self.layers = layers

        self.net_u = tf.keras.Sequential()
        self.net_u.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
        for i in range(1, len(self.layers) - 1):
            self.net_u.add(
                tf.keras.layers.Dense(self.layers[i], activation=tf.nn.tanh, kernel_initializer="glorot_normal"))
        self.net_u.add(tf.keras.layers.Dense(self.layers[-1], activation=None, kernel_initializer="glorot_normal"))

        self.tf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.k = 0
        self.loss_array = np.array([])
        self.loss_weak_array = np.array([])
        self.loss_contact_array = np.array([])
        self.epochs = 0
        self.sizes_w = []
        self.sizes_b = []
        self.w_c = args.w_c
        self.force = args.force
        for i, width in enumerate(self.layers):
            if i != 1:
                self.sizes_w.append(int(width * self.layers[1]))
                self.sizes_b.append(int(width if i != 0 else self.layers[1]))

    def wrap_training_variables(self):
        var = self.net_u.trainable_variables
        return var

    @tf.function
    def loss(self, X_f, X_contact, X_out):
        nb_colloc = 2500
        nb_contact = 600
        f_final = 0
        f_weak = 0
        f_contact_all = 0
        for i in range(args.k_pneu):
            f = self.net_f(X_f[i * nb_colloc:(i + 1) * nb_colloc])
            # f_volume = self.net_volume(X_f[i * nb_colloc:(i + 1) * nb_colloc])
            f_contact = self.net_contact_ineq(X_contact)
            loss_weak = (f)
            loss_contact = tf.reduce_sum(tf.square(f_contact))
            #                tf.reduce_mean(ineq1_right) + tf.reduce_mean(ineq2_right) + tf.reduce_mean(tf.square(eq3_right))
            loss = loss_weak + loss_contact*self.w_c
            f_final += loss
            f_weak += loss_weak
            f_contact_all += loss_contact
        return f_final, f_weak, f_contact_all

    @tf.function
    def net_bound(self, X_temp):
        mu = self.mu
        lbda = self.lbda
        x_bound = X_temp[:, 0:1]
        y_bound = X_temp[:, 1:2]
        z_bound = X_temp[:, 2:-2]
        r1 = X_temp[:, -2:-1]
        cy = X_temp[:, -1:X_temp.shape[1]]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_bound)
            tape.watch(y_bound)
            tape.watch(z_bound)
            X_bound = tf.concat([x_bound, y_bound, z_bound], axis=1)
            dist = (x_bound - 0) ** 2 / (r1/L0) ** 2 + (y_bound - cy/L0) ** 2 / (r1/L0) ** 2 - 1
            u = self.net_u(X_bound)[:, 0:1] * dist/6
            v = self.net_u(X_bound)[:, 1:2] * dist/6 - 0.01

        del tape
        f_bound = self.force * (v)
        return tf.reduce_mean(f_bound)

    @tf.function
    def net_contact_ineq(self, X_temp):
        mu = self.mu
        lbda = self.lbda
        param = 0.9
        x_f = X_temp[:, 0:1]
        y_f = X_temp[:, 1:2]
        z_f = X_temp[:, 2:-2]
        r1 = X_temp[:, -2:-1]
        cy = X_temp[:, -1:X_temp.shape[1]]
        #         d = solve(r1,r2,X_temp)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_f)
            tape.watch(y_f)
            tape.watch(z_f)
            X_f = tf.concat([x_f, y_f, z_f], axis=1)
            dist = (x_f - 0) ** 2 / (r1/L0) ** 2 + (y_f - cy/L0) ** 2 / (r1/L0) ** 2 - 1
            u = self.net_u(X_f)[:, 0:1] * dist / 6
            v = self.net_u(X_f)[:, 1:2] * dist / 6 - 0.01

            def_x_f = x_f + u
            def_y_f = y_f + v

            x_f_ori = def_x_f - u
            y_f_ori = def_y_f - v

            u_x = tape.gradient(u, x_f) * tape.gradient(x_f_ori, def_x_f)  # /tape.gradient(def_x_f, x_f)
            u_y = tape.gradient(u, y_f) * tape.gradient(y_f_ori, def_y_f)
            v_x = tape.gradient(v, x_f) * tape.gradient(x_f_ori, def_x_f)
            v_y = tape.gradient(v, y_f) * tape.gradient(y_f_ori, def_y_f)

        N_x = 0  # x_f - cx / L0
        N_y = -1  # y_f - cy / L0
        n_x = 0  # N_x / tf.math.sqrt(N_x ** 2 + N_y ** 2)
        n_y = -1  # N_y / tf.math.sqrt(N_x ** 2 + N_y ** 2)

        u_n = u * n_x + v * n_y
        u_T_x = u - u_n * n_x
        u_T_y = v - u_n * n_y
        u_T = tf.stack([u_T_x, u_T_y], axis=1)

        F_11 = 1 + u_x
        F_12 = u_y
        F_21 = v_x
        F_22 = 1 + v_y

        E_11 = 0.5 * (u_x ** 2 + v_x ** 2 + 2 * u_x)
        E_22 = 0.5 * (u_y ** 2 + v_y ** 2 + 2 * v_y)
        E_12 = 0.5 * (u_x * u_y + v_x * v_y + u_y + v_x)
        E_21 = E_12

        S_11 = lbda * (E_11 + E_22) + 2 * mu * E_11
        S_22 = lbda * (E_11 + E_22) + 2 * mu * E_22
        S_12 = 2 * mu * E_12
        S_21 = 2 * mu * E_21

        Pi_11 = F_11 * S_11 + F_12 * S_21
        Pi_12 = F_11 * S_12 + F_12 * S_22
        Pi_21 = F_21 * S_11 + F_22 * S_21
        Pi_22 = F_21 * S_12 + F_22 * S_22

        sigma_n = n_x * (Pi_11 * n_x + Pi_12 * n_y) + n_y * (Pi_21 * n_x + Pi_22 * n_y)
        sigma_T_x = Pi_11 * n_x + Pi_12 * n_y - sigma_n * n_x
        sigma_T_y = Pi_21 * n_x + Pi_22 * n_y - sigma_n * n_y

        sigma_T = tf.stack([sigma_T_x, sigma_T_y], axis=1)

        g = -y_f * n_y

        f_contact = -(u_n - g) - sigma_n - tf.math.sqrt((u_n - g) ** 2 + sigma_n ** 2)

        return f_contact

    @tf.function
    def net_f(self, X_temp):
        mu = self.mu
        lbda = self.lbda
        x_f = X_temp[:, 0:1]
        y_f = X_temp[:, 1:2]
        z_f = X_temp[:, 2:]
        r1 = X_temp[:, -2:-1]
        cy = X_temp[:, -1:X_temp.shape[1]]
        #         d = solve(r1,r2,X_temp)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_f)
            tape.watch(y_f)
            tape.watch(z_f)
            X_f = tf.concat([x_f, y_f, z_f], axis=1)
            dist = (x_f - 0) ** 2 / (r1/L0) ** 2 + (y_f - cy/L0) ** 2 / (r1/L0) ** 2 - 1
            u = self.net_u(X_f)[:, 0:1] * dist / 6
            v = self.net_u(X_f)[:, 1:2] * dist / 6 - 0.01

            def_x_f = x_f + u
            def_y_f = y_f + v

            x_f_ori = def_x_f - u
            y_f_ori = def_y_f - v

            u_x = tape.gradient(u, x_f) * tape.gradient(x_f_ori, def_x_f)
            u_y = tape.gradient(u, y_f) * tape.gradient(y_f_ori, def_y_f)
            v_x = tape.gradient(v, x_f) * tape.gradient(x_f_ori, def_x_f)
            v_y = tape.gradient(v, y_f) * tape.gradient(y_f_ori, def_y_f)

        f = 0.5 * lbda * 1 / 4 * (u_x ** 2 + u_y ** 2 + v_x ** 2 + v_y ** 2 + 2 * u_x + 2 * v_y) ** 2 + mu * 1 / 4 * (
                (u_x ** 2 + v_x ** 2 + 2 * u_x) ** 2 + (u_y ** 2 + v_y ** 2 + 2 * v_y) ** 2 + 2 * (
                u_x * u_y + v_x * v_y + u_y + v_x) ** 2)
        return tf.reduce_mean(f)

    @tf.function
    def get_grad(self, X_f, X_contact, X_out):

        with tf.GradientTape() as tape:
            loss_value, loss_weak, loss_contact = self.loss(X_f, X_contact, X_out)

        grads = tape.gradient(loss_value, self.wrap_training_variables())

        return loss_value, loss_weak, loss_contact, grads

    def train(self, nb_epochs_supervised=0, nb_epochs_pinns=0, nb_epochs_pinns_e4=0, nb_epochs_pinns_e5=0,
              lbfgs_supervised=False, lbfgs_pinns=False):

        @tf.function
        def train_step(X_f, X_contact, X_out):
            loss_value, loss_weak, loss_contact, grads = self.get_grad(X_f, X_contact, X_out)
            self.tf_optimizer.apply_gradients(
                zip(grads, self.net_u.trainable_variables))
            return loss_value, loss_weak, loss_contact

        def callback(x=None):
            if self.epochs % 1 == 0:
                print('Loss pinns at epoch %d (L-BFGS-B):' % self.epochs, self.current_loss)

            # self.hist.append(self.current_loss)
            self.epochs += 1
        def optimizer_lbfgs(X_f, X_contact, X_out, method='L-BFGS-B', **kwargs):
            """
            Optimizer LBFGS to minimize the loss

            :param X_f: Collocation points
            :type X_f: numpy.ndarray
            :param param_f: PDE parameters
            :type param_f: numpy.ndarray
            :meta private:
            """
            def get_weight():
                list_weight = []
                for variable in self.wrap_training_variables():
                    list_weight.extend(variable.numpy().flatten())
                list_weight = tf.convert_to_tensor(list_weight)
                return list_weight

            def set_weight(list_weight):
                index = 0
                for variable in self.wrap_training_variables():
                    if len(variable.shape) == 2:
                        len_weights = variable.shape[0] * variable.shape[1]
                        new_variable = tf.reshape(list_weight[index:index + len_weights], (variable.shape[0], variable.shape[1]))
                        index += len_weights
                    elif len(variable.shape) == 1:
                        len_biases = variable.shape[0]
                        new_variable = list_weight[index:index + len_biases]
                        index += len_biases
                    else:
                        new_variable = list_weight[index]
                        index += 1
                    variable.assign(tf.cast(new_variable, 'float64'))

            def get_loss_and_grad(w):
                set_weight(w)
                loss_value, loss_weak, loss_contact, grad = self.get_grad(X_f, X_contact, X_out)
                self.loss_array = np.append(self.loss_array, loss_value.numpy())
                self.loss_contact_array = np.append(self.loss_contact_array, loss_contact.numpy())
                self.loss_weak_array = np.append(self.loss_weak_array, loss_weak.numpy())
                loss = loss_value.numpy().astype(np.float64)
                self.current_loss = loss


                grad_flat = []
                for g in grad:
                    grad_flat.extend(g.numpy().flatten())
                grad_flat = np.array(grad_flat, dtype=np.float64)
                return loss, grad_flat

            return scipy.optimize.minimize(fun=get_loss_and_grad,
                                           x0=get_weight(),
                                           jac=True,
                                           method=method, callback=callback, **kwargs)


        for epoch in range(nb_epochs_pinns):
            loss_value, loss_weak, loss_contact = train_step(self.X_f, self.X_contact, self.X_out)
            print('Loss pinns at %d epoch (Adam):' % epoch, loss_value.numpy())
            self.loss_array = np.append(self.loss_array, loss_value.numpy())

        optimizer_lbfgs(self.X_f, self.X_contact, self.X_out,
                                  method='L-BFGS-B',
                                  options={'maxiter': 1000000,
                                           'maxfun': 1000000,
                                           'maxcor': 100,
                                           'maxls': 100,
                                           'ftol': 0,
                                           'gtol': 1.0 * np.finfo(float).eps})


model = GADEM_toytire_SVK(X_colloc, contact_line, contact_line, layers)
model.train(0, int(args.nb_epochs_total), 0, 0)

