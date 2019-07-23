import os
import sys
from enum import Enum

class Values(Enum):
    LABELS_1 = 'src_labels'
    LABELS_2 = 'tgt_labels'
    CATEGORIES = 'category'
    CLASS = 'class'
    PROPERTY = 'property'

class InternalProperties:
    def __init__(self, data):
        assert type(data) == dict, "Properties must be provided as a dict to the InternalProperties class"
        assert Values.LABELS_1.value in data.keys(), "Source Label-properties must be provided as 'labels'-entry in the dict of the InternalProperties"
        assert Values.LABELS_2.value in data.keys(), "Target Label-properties must be provided as 'labels'-entry in the dict of the InternalProperties"
        assert Values.CATEGORIES.value in data.keys(), "Category-property must be provided as 'category'-entry in the dict of the InternalProperties"
        assert Values.CLASS.value in data.keys(), "Class-property must be provided as 'category'-entry in the dict of the InternalProperties"
        assert Values.PROPERTY.value in data.keys(), "Property-property must be provided as 'category'-entry in the dict of the InternalProperties"
        assert len(data.keys()) == 5, "Properties-definition must contain exactly 5 entries"
        assert type(data[Values.LABELS_1.value]) == list, "Target Labels must be provided as a list in the dict of the InternalProperties"
        assert type(data[Values.LABELS_2.value]) == list, "Source Labels must be provided as a list in the dict of the InternalProperties"
        assert type(data[Values.CATEGORIES.value]) == str, "Categories must be provided as a string in the dict of the InternalProperties"
        assert type(data[Values.CLASS.value]) == str, "Class must be provided as a string in the dict of the InternalProperties"
        assert type(data[Values.PROPERTY.value]) == str, "Property must be provided as a string in the dict of the InternalProperties"

        self.src_label_properties = data[Values.LABELS_1.value]
        self.tgt_label_properties = data[Values.LABELS_2.value]
        self.class_descriptor = data[Values.CLASS.value]
        self.property_descriptor = data[Values.PROPERTY.value]
        self.category_property = data[Values.CATEGORIES.value]

        # Register cache files
