# import random
# import unittest

# from memory.utils import (argmax, get_duplicate_dicts, list_duplicates_of,
#                           read_json, read_yaml, seed_everything, write_json,
#                           write_yaml)


# class UtilsTest(unittest.TestCase):
#     def test_seed_everything(self):
#         seed_everything(random.randint(0, 1000000))

#     def test_argmax(self):
#         self.assertEqual(argmax([6, 1, 2, 3, 4, 5]), 0)

#     def test_get_duplicate_dicts(self):
#         foo = get_duplicate_dicts({"foo": 1}, [{"foo": 1}, {"bar": 2}, {"foo": 1}])
#         self.assertEqual(foo, [{"foo": 1}, {"foo": 1}])

#     def test_list_duplicates_of(self):
#         foo = list_duplicates_of(
#             [{"foo": 1}, {"bar": 2}, {"foo": 2}, {"foo": 1}], {"foo": 1}
#         )
#         self.assertEqual(foo, [0, 3])
