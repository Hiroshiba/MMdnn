from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import six
from conversion_imagenet import TestModels


def get_test_table():
    return {
        'keras': {
            'music_tagger_crnn': [TestModels.pytorch_emit],
        }}


def test_keras():
    test_table = get_test_table()
    tester = TestModels(test_table)
    tester._test_function('keras', tester.keras_parse)


if __name__ == '__main__':
    test_keras()
