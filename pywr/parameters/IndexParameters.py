import numpy as np
from scipy.interpolate import Rbf
from .parameters import Parameter, load_parameter, load_parameter_values, IndexParameter
from ..nodes import Storage


class RectifierParameter(Parameter):

    def __init__(self, model, value, lower_bounds=0.0, upper_bounds=np.inf, **kwargs):
        super(RectifierParameter, self).__init__(model, **kwargs)
        self._value = value
        self.double_size = 1
        self.integer_size = 0
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

    def calc_values(self, timestep):
        # constant parameter can just set the entire array to one value
        if self._value < 0.0:
            self.__values[...] =  0.0
        else:
            self.__values[...] = (self._upper_bounds -  self._lower_bounds) * self._value + self._lower_bounds

    def value(self, ts, scenario_index):
        return self._value

    def set_double_variables(self, values):
        self._value = values[0]

    def get_double_variables(self):
        return np.array([self._value, ], dtype=np.float64)

    def get_double_lower_bounds(self):
        return np.array([-0.75], dtype=np.float64)

    def get_double_upper_bounds(self):
        return np.array([1.0], dtype=np.float64)

    @classmethod
    def load(cls, model, data):
        if "value" in data:
            value = data.pop("value")
        else:
            value = load_parameter_values(model, data)
        parameter = cls(model, value, **data)
        return parameter


RectifierParameter.register()


class IndexVariableParameter(IndexParameter):

    def __init__(self, model, value, lower_bounds=0, upper_bounds=1, **kwargs):
        super(IndexVariableParameter, self).__init__(model, **kwargs)
        self._value = round(value)
        self.integer_size = 1
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

    def calc_values(self, timestep):
        # constant parameter can just set the entire array to one value
        self.__indices[...] = self._value
        self.__values[...] = self._value

    def set_integer_variables(self, values):
        self._value  = values[0]

    def get_integer_variables(self):
        return np.array([self._value, ], dtype=np.int32)

    def get_integer_lower_bounds(self):
        return np.array([self._lower_bounds, ], dtype=np.int32)

    def get_integer_upper_bounds(self):
        return np.array([self._upper_bounds, ], dtype=np.int32)

    @classmethod
    def load(cls, model, data):
        if "value" in data:
            value = data.pop("value")
        else:
            value = load_parameter_values(model, data)
        parameter = cls(model, value, **data)
        return parameter


IndexVariableParameter.register()
