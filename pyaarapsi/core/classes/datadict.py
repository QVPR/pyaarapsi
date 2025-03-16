#! /usr/bin/env python3
'''
DataDict object.
'''
from __future__ import annotations

from typing import List, Any

class DataDict(dict):
    '''
    Simple extension to dictionaries in order to permit fast width-wise data appending
    '''
    def __init__(self, *args, **kwargs):
        super(DataDict, self).__init__(*args, **kwargs)

        self._columns = []

    def addcolumns(self, columns: List) -> DataDict:
        '''
        Create a stack of columns
        '''
        assert isinstance(columns, list)
        assert len(self._columns) == 0, "add_columns can only be used once!"
        self._columns.extend(columns)
        for column in columns:
            self[column] = []
        return self

    def append(self, data: List) -> DataDict:
        '''
        Append data to existing columns
        '''
        assert isinstance(data, list)
        assert len(data) == len(self._columns)
        for datum, column in zip(data, self._columns):
            self[column].append(datum)
        return self

    def getdata(self) -> List[list]:
        '''
        Get out data
        '''
        return [self[column] for column in self._columns]

    def getcolumns(self) -> List[Any]:
        '''
        Get assigned column values
        '''
        return self._columns
