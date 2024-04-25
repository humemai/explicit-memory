import random
import unittest

from humemai.utils import *


class UtilsTest(unittest.TestCase):
    def test_seed_everything(self):
        seed_everything(random.randint(0, 1000000))

    def test_argmax(self):
        self.assertEqual(argmax([6, 1, 2, 3, 4, 5]), 0)

    def test_get_duplicate_dicts(self):
        foo = get_duplicate_dicts({"foo": 1}, [{"foo": 1}, {"bar": 2}, {"foo": 1}])
        self.assertEqual(foo, [{"foo": 1}, {"foo": 1}])

    def test_list_duplicates_of(self):
        foo = list_duplicates_of(
            [{"foo": 1}, {"bar": 2}, {"foo": 2}, {"foo": 1}], {"foo": 1}
        )
        self.assertEqual(foo, [0, 3])

    def test_remove_possessive(self):
        foo = remove_posession("John's book")
        self.assertEqual(foo, "book")
