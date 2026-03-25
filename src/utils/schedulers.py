import math
import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Scheduler:

    def __init__(self):
        self.counter = 0
        self.variables = []

    def rest_impl(self):
        raise NotImplementedError

    def reset(self):
        self.rest_impl()
        self.counter = 0

    def step_impl(self):
        raise NotImplementedError

    def step(self):
        self.step_impl()
        self.counter += 1


class SchedulerLinear(Scheduler):

    def __init__(
            self,
            name,
            iter_start, 
            iter_end,
            val_start=0,
            val_end=1,
            ):
        super().__init__()
        self.name = name
        self.variables.append(name)
        self.iter_start = iter_start
        self.iter_end = iter_end
        self.val_start = val_start
        self.val_end = val_end
        self.reset()
        
    def rest_impl(self):
        setattr(self, self.name, self.val_start)

    def step_impl(self):
        p = (self.counter - self.iter_start) / (self.iter_end - self.iter_start)
        p = min(1, max(0, p))
        val = (1 - p) * self.val_start + p * self.val_end
        setattr(self, self.name, val)


class SchedulerSigmoid(Scheduler):

    def __init__(
            self, 
            name,
            iter_start, 
            iter_end,
            val_start=0,
            val_end=1,
            ):
        super().__init__()
        self.name = name
        self.variables.append(name)
        self.iter_start = iter_start
        self.iter_end = iter_end
        self.val_start = val_start
        self.val_end = val_end
        self.reset()

    def rest_impl(self):
        setattr(self, self.name, self.val_start)
        
    def step_impl(self):


        if self.counter < self.iter_start:
            p = 0
        elif self.iter_end < self.counter:
            p = 1
        else:
            p = (self.counter - self.iter_start) / (self.iter_end - self.iter_start)
            min_sigmoid = -5 
            max_sigmoid = 5
            p = sigmoid(min_sigmoid + p * (max_sigmoid - min_sigmoid))

        val = (1 - p) * self.val_start + p * self.val_end
        setattr(self, self.name, val)


class SchedulerCombined:

    def __init__(self, *schedulers):
        super().__init__()
        self.schedulers = schedulers
        unique_names = set()
        for sch in self.schedulers:
            for name in sch.variables:
                assert name not in unique_names
                unique_names.add(name)
        self.variables = list(unique_names)
        self.reset()

    def update(self):
        for sch in self.schedulers:
            for name in sch.variables:
                value = getattr(sch, name)
                setattr(self, name, value)

    def reset(self):
        for sch in self.schedulers:
            sch.reset()
        self.update()
        self.counter = 0
        
    def step(self):
        for sch in self.schedulers:
            sch.step()
        self.update()
        self.counter += 1


def scheduler_combined_two(scheduler1=None, scheduler2=None):
    return SchedulerCombined(scheduler1, scheduler2)

def scheduler_combined_three(scheduler1=None, scheduler2=None, scheduler3=None):
    return SchedulerCombined(scheduler1, scheduler2, scheduler3)