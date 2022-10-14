# -*- coding: utf-8 -*-
"""Deep One-Class Classification for outlier detection
"""
# Author: Rafal Bodziony <bodziony.rafal@gmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from pyod.utils.utility import check_parameter

#from pyod.base import BaseDetector
from pyod.models.deep_svdd import DeepSVDD as BaseDetector
#from pyod.base_dl import _get_tensorflow_version

# if tensorflow 2, import from tf directly
if True:#_get_tensorflow_version() == 2:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras import Model, Input
else:
    raise ModuleNotFoundError('DeepSVDD runs only with TensorFlow 2.0+')

from tensorflow.keras import backend as K


class DeepSVDD(BaseDetector):

    def __init__(self, c=None,
                 use_ae=False,
                 hidden_neurons=None,
                 hidden_activation='relu',
                 output_activation='sigmoid',
                 optimizer='adam',
                 epochs=100, batch_size=32, dropout_rate=0.2,
                 use_bias=False,
                 l2_regularizer=0.1, validation_size=0.1, preprocessing=True,
                 verbose=1, random_state=None, contamination=0.1):
        super(DeepSVDD, self).__init__(contamination=contamination)
        self.c = c
        self.use_ae = use_ae
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.use_bias=use_bias
        self.l2_regularizer = l2_regularizer
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.random_state = random_state

        if self.random_state is not None:
            tf.random.set_seed(self.random_state)
        # default values
        if self.hidden_neurons is None:
            self.hidden_neurons = [64, 32]

        self.hidden_neurons_ = self.hidden_neurons

        check_parameter(dropout_rate, 0, 1, param_name='dropout_rate',
                        include_left=True)

    def _init_c(self, X_norm, eps=0.1):
        # create true Center value from model predict of intermediate layers
        model_center = Model(self.model_.inputs,
                             self.model_.get_layer('net_output').output)

        out_ = model_center.predict(X_norm)
        nf_predict = out_.shape[0]
        out_ = np.sum(out_, axis=0)
        out_ /= nf_predict
        self.c = out_
        self.c[(abs(self.c) < eps) & (self.c < 0)] = -eps
        self.c[(abs(self.c) < eps) & (self.c > 0)] = eps

        return self

    def _build_model(self, training=True):

        inputs = Input(shape=(self.n_features_,))
        x=inputs
        lays=[]
        lays.append(Dense(self.hidden_neurons_[0], activation=self.hidden_activation,use_bias=self.use_bias,
                  activity_regularizer=l2(self.l2_regularizer)))
        for hidden_neurons in self.hidden_neurons_[1:-1]:
            lays.append(Dense(hidden_neurons, activation=self.hidden_activation,use_bias=self.use_bias,
                      activity_regularizer=l2(self.l2_regularizer)))
            lays.append(Dropout(self.dropout_rate))

        # add name to last hidden layer
        lays.append(Dense(self.hidden_neurons_[-1], activation=self.hidden_activation,use_bias=self.use_bias,
                  activity_regularizer=l2(self.l2_regularizer),
                  name='net_output'))

        self.lays=lays
        for lay in lays:
            x = lay(x)


        self.pmodel=Model(inputs,x)

        # build distance loss
        dist = K.sum((x - self.c) ** 2, axis=-1)#*float(x.shape[-1])
        outputs = dist
        loss = K.mean(dist)

        # Instantiate Deep SVDD
        dsvd = Model(inputs, outputs)

        # Weight decay
        w_d = 1e-6 * sum([np.linalg.norm(w) for w in dsvd.get_weights()])

        # Use AutoEncoder version of DeepSVDD
        if self.use_ae:
            for reversed_neurons in self.hidden_neurons_[::-1]:
                x = Dense(reversed_neurons, activation=self.hidden_activation,use_bias=self.use_bias,
                          activity_regularizer=l2(self.l2_regularizer))(x)
                x = Dropout(self.dropout_rate)(x)
            x = Dense(self.n_features_, activation=self.output_activation,use_bias=self.use_bias,
                      activity_regularizer=l2(self.l2_regularizer))(x)
            dsvd.add_loss(
                loss + K.mean(K.square(x - inputs)) + w_d)
        else:
            dsvd.add_loss(loss + w_d)

        dsvd.compile(optimizer=self.optimizer)

        if self.verbose >= 1 and training:
            print(dsvd.summary())
        return dsvd

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        # Verify and construct the hidden units
        self.n_samples_, self.n_features_ = X.shape[0], X.shape[1]

        # Standardize data for better performance
        if self.preprocessing:
            self.scaler_ = StandardScaler()
            X_norm = self.scaler_.fit_transform(X)
        else:
            X_norm = np.copy(X)

        # Shuffle the data for validation as Keras do not shuffling for
        # Validation Split
        np.random.shuffle(X_norm)

        # Validate and complete the number of hidden neurons
        if np.min(self.hidden_neurons) > self.n_features_ and self.use_ae:
            raise ValueError("The number of neurons should not exceed "
                             "the number of features")
        if self.c is None:
            self.c = 0.0
            self.model_ = self._build_model(training=False)
            self._init_c(X_norm)

        # Build DeepSVDD model & fit with X
        self.model_ = self._build_model()
        self.history_ = self.model_.fit(X_norm, X_norm,
                                        epochs=self.epochs,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        validation_split=self.validation_size,
                                        verbose=self.verbose).history
        # Predict on X itself and calculate the reconstruction error as
        # the outlier scores. Noted X_norm was shuffled has to recreate
        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        self.decision_scores_ = self.model_.predict(X_norm)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['model_', 'history_'])
        X = check_array(X)

        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        # Predict on X and return the reconstruction errors
        pred_scores = self.model_.predict(X_norm)
        return pred_scores
