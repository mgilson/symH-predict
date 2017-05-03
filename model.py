"""RNN Model building blocks."""
import json
import os

from keras.models import Sequential, load_model
from keras.layers.core import Activation, Dense, Dropout

from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import LambdaCallback


class ModelWrapper(object):
    """A wrapper around a keras sequential model to allow saving/restoring.

    Unlike a normal keras save/restore, this also saves loss and val_loss.
    """
    def __init__(self, model, name=None, save_path='.', save_freq=10):
        self.model = model
        self.name = name
        self.loss = []
        self.val_loss = []
        self.epoch = 0
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_epoch_end(self, epoch, logs=None):
        if not logs:
            logs = {}
        self.epoch = epoch
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        if self.save_freq and (self.epoch + 1) % self.save_freq == 0:
            self.save(path=self.save_path)

    def fit(self, *args, **kwargs):
        """Wrapper around "fit" method."""
        callbacks = kwargs.setdefault('callbacks', [])
        callbacks.append(LambdaCallback(on_epoch_end=self._on_epoch_end))
        return self.model.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        """Wrapper around evaluate method."""
        return self.model.evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """Wrapper around predict method."""
        return self.model.predict(*args, **kwargs)

    def compile(self, *args, **kwargs):
        """Wrapper around compile method."""
        return self.model.compile(*args, **kwargs)

    def save(self, path='.'):
        """Save the model and training meta-data."""
        if not self.name:
            raise RuntimeError('Must have a name to save model.')
        fname = os.path.join(path, self.name + '.%s.json' % self.epoch)
        with open(fname, 'w') as outfile:
            json.dump({
                'name': self.name,
                'loss': self.loss,
                'val_loss': self.val_loss,
                'epoch': self.epoch,
                'save_path': self.save_path,
            }, outfile)
        fname = os.path.join(path, self.name + '.%s.h5' % self.epoch)
        self.model.save(fname)

    @classmethod
    def restore(cls, fname):
        """Restore a model."""
        self = cls(None)
        with open(fname, 'r') as outfile:
            vars(self).update(json.load(outfile))
        base, _ = os.path.splitext(outfile)
        fname_h5 = base + '.h5'
        self.model = load_model(fname_h5)
        return self


def build_model(  # pylint: disable=too-many-arguments
        input_shape,
        n_points_predict,
        hidden=64,
        rnn_type=GRU,
        stacks=1,
        dropout=.2,
        activation='linear'):
    """Build a Keras Sequential model.

    Args:
        input_shape (tuple): The input shape for the input layer.
        n_points_predict (number): The number of points to predict into the
            future.
        hidden (number): The number of units in the RNN nodes.
        rnn_type (type): A Keras RNN layer.  Either keras.layers.recurrent.LSTM,
            or keras.layers.recurrent.GRU
        stacks (int): The number of RNN layers to use.
        dropout (float): the value for the dropout layer.
        activation: The activation function.
    Returns:
        keras.Sequential.
    """

    model = Sequential()

    kwargs = {
        'input_shape': input_shape,
    }
    if stacks > 1:
        kwargs['return_sequences'] = True
    model.add(rnn_type(hidden, **kwargs))

    del kwargs['input_shape']
    for i in range(2, stacks+1):
        if i == stacks:
            del kwargs['return_sequences']
        model.add(rnn_type(hidden, **kwargs))

    model.add(Dropout(dropout))
    model.add(Dense(n_points_predict))
    if activation:
        model.add(Activation(activation))
    return model


def compile_model(model, loss='mse', optimizer='rmsprop', **kwargs):
    """Compile the model.

    This just wraps model.compile with some convenient defaults.
    """

    model.compile(loss=loss, optimizer=optimizer, **kwargs)
