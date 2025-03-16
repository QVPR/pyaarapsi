#!/usr/bin/env python3
'''
Class definition for a helper timer (for debugging)
'''
import time
from typing import Optional, List, Callable
from typing_extensions import Self

class Timer:
    '''
    Helper timer class for measuring and logging execution times
    '''
    def __init__(self, rospy_on: bool = False, precision: int = 5) -> Self:
        self.points = []
        self.rospy_on = rospy_on
        self.add_bounds = False
        self.precision = precision
        self.threshold = 10**(-1 * self.precision)
    #
    def add(self) -> Self:
        '''
        Add a measuring point
        '''
        self.points.append(time.perf_counter())
        return self
    #
    def addb(self) -> Self:
        '''
        At the end, append the bounded time elapsed
        '''
        self.add_bounds = True
        return self
    #
    def calc(self) -> List[float]:
        '''
        Calculate times between measuring points
        '''
        times = []
        for i in range(len(self.points) - 1):
            this_time = abs(self.points[i+1]-self.points[i])
            if this_time < self.threshold:
                this_time = 0.0
            times.append(this_time)
        if self.add_bounds and len(self.points) > 0:
            times.append(abs(self.points[-1] - self.points[0]))
        return times
    #
    def show(self, name: Optional[str] = None) -> Self:
        '''
        Print result
        '''
        times = self.calc()
        string = str([("%" + str(int(4 + self.precision)) + "." \
                       + str(int(self.precision)) + "f") % i for i in times]).replace(' ','')
        if name is not None:
            string = "[" + name + "] " + string
        self.print(string)
        self.clear()
        return self
    #
    def clear(self) -> Self:
        '''
        Clear all measured points, reset add_bounds
        '''
        self.points[:] = []
        self.add_bounds = False
        return self
    #
    def print(self, string: str) -> Self:
        '''
        Print helper
        '''
        print(string)
        return self
    #
    def time_function(self, func: Callable) -> float:
        '''
        Time a function
        '''
        self.clear()
        self.add()
        func()
        self.add()
        diff = self.points[1] - self.points[0]
        self.clear()
        return diff
