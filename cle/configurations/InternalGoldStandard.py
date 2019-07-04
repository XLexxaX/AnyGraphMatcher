import os
import sys
from enum import Enum

class Values(Enum):
    TRAINSETS = 'trainsets'
    TESTSETS = 'testsets'

class InternalGoldStandard:
    def __init__(self, data):
        assert type(data) == dict, "Train and test set must be provided as a dict to the InternalGoldStandard class"
        assert Values.TRAINSETS.value in data.keys(), "Train set must be provided as 'trainset'-entry in the dict of the InternalGoldStandard"
        #assert 'testsets' in data.keys(), "Test set must be provided as 'trainset'-entry in the dict of the InternalGoldStandard"
        assert len(data.keys()) == 1 or len(data.keys()) == 2, "Gold standard must contain exactly 1 or 2 entries: Trainset(, testset)"
        assert type(data[Values.TRAINSETS.value]) == list, "Train set must be provided as a list in the dict of the InternalGoldStandard"
        self.raw_testsets = list()
        if Values.TESTSETS.value in data.keys():
            assert type(data[Values.TESTSETS.value]) == list, "Test set must be provided as a list in the dict of the InternalGoldStandard"
            self.raw_testsets = data[Values.TESTSETS.value]

        self.prepared_testsets = list()
        self.raw_trainsets = data[Values.TRAINSETS.value]
        self.prepared_trainsets = list()

        # Register cache files

