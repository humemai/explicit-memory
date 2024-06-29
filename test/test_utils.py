import os
import unittest
from tempfile import NamedTemporaryFile
from unittest.mock import patch

from humemai.utils import (argmax, get_duplicate_dicts, list_duplicates_of,
                           load_questions, merge_lists, read_data, read_json,
                           read_pickle, read_yaml, remove_timestamp,
                           write_json, write_pickle, write_yaml)


class TestUtils(unittest.TestCase):

    def test_remove_timestamp(self):
        entry = ["Alice", "likes", "Bob", {"timestamp": 12345}]
        result = remove_timestamp(entry)
        self.assertEqual(result, ["Alice", "likes", "Bob"])

    def test_read_json_write_json(self):
        data = {"key": "value"}

        # Test write_json
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_filename = temp_file.name
            write_json(data, temp_filename)
            self.assertTrue(os.path.exists(temp_filename))

        # Test read_json
        read_data = read_json(temp_filename)
        self.assertEqual(read_data, data)

        os.remove(temp_filename)

    def test_read_yaml_write_yaml(self):
        data = {"key": "value"}

        # Test write_yaml
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_filename = temp_file.name
            write_yaml(data, temp_filename)
            self.assertTrue(os.path.exists(temp_filename))

        # Test read_yaml
        read_data = read_yaml(temp_filename)
        self.assertEqual(read_data, data)

        os.remove(temp_filename)

    def test_write_pickle_read_pickle(self):
        data = {"key": "value"}

        # Test write_pickle
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_filename = temp_file.name
            write_pickle(data, temp_filename)
            self.assertTrue(os.path.exists(temp_filename))

        # Test read_pickle
        read_data = read_pickle(temp_filename)
        self.assertEqual(read_data, data)

        os.remove(temp_filename)

    def test_read_data(self):
        data_path = "test_data.json"
        test_data = {"train": [1, 2, 3], "val": [4, 5, 6], "test": [7, 8, 9]}

        # Mocking read_json function
        with patch("humemai.utils.read_json", return_value=test_data):
            result = read_data(data_path)

        self.assertEqual(result, test_data)

    def test_load_questions(self):
        questions_path = "test_questions.json"
        test_questions = {"question1": "answer1", "question2": "answer2"}

        # Mocking read_json function
        with patch("humemai.utils.read_json", return_value=test_questions):
            result = load_questions(questions_path)

        self.assertEqual(result, test_questions)

    def test_argmax(self):
        iterable = [5, 8, 2, 10, 3]
        result = argmax(iterable)
        self.assertEqual(result, 3)

    def test_get_duplicate_dicts(self):
        search_dict = {"key1": "value1"}
        target_list = [
            {"key1": "value1", "key2": "value2"},
            {"key1": "value1", "key2": "value3"},
        ]
        expected_duplicates = [
            {"key1": "value1", "key2": "value2"},
            {"key1": "value1", "key2": "value3"},
        ]

        result = get_duplicate_dicts(search_dict, target_list)
        self.assertEqual(result, expected_duplicates)

    def test_list_duplicates_of(self):
        seq = [1, 2, 3, 2, 4, 5, 2]
        item = 2
        expected_duplicates = [1, 3, 6]

        result = list_duplicates_of(seq, item)
        self.assertEqual(result, expected_duplicates)

    def test_merge_lists(self):
        lists = [
            ["Alice", "likes", "Bob", {"current_time": 1}],
            ["Alice", "likes", "Bob", {"timestamp": [12345]}],
            ["Alice", "loves", "Charlie", {"strength": 2}],
        ]

        expected_merged_list = [
            ["Alice", "likes", "Bob", {"current_time": 1, "timestamp": [12345]}],
            ["Alice", "loves", "Charlie", {"strength": 2}],
        ]

        result = merge_lists(lists)

        self.assertEqual(
            len(result), len(expected_merged_list)
        )  # Ensure both lists have the same length

        for expected_item in expected_merged_list:
            self.assertIn(
                expected_item, result
            )  # Check if each expected item is in the result
