#!/usr/bin/env python3

"""
The benchmark modules provides a convenient interface to standardized benchmarks in the literature.
It provides train/validation/test TaskDatasets and TaskTransforms for pre-defined datasets.

This utility is useful for researchers to compare new algorithms against existing benchmarks.
For a more fine-grained control over tasks and data, we recommend directly using `l2l.data.TaskDataset` and `l2l.data.TaskTransforms`.
"""

import os
import learn2learn as l2l
import random
import pandas as pd
from collections import namedtuple
import torch
from PIL import Image

BenchmarkTasksets = namedtuple(
    'BenchmarkTasksets', ('train', 'validation', 'test_adaptation', 'test_evaluation'))


class TensorTask():
    def __init__(self, data_directory):
        self.data_directory = data_directory
        filename_tasks = self._read_json()
        self.tensor_tasks = self._to_tensor_tasks(filename_tasks)
        # tensor_tasks [data, labels]

    def sample(self):
        tensor_task = random.choice(self.tensor_tasks)
        return tensor_task['images'], tensor_task['labels']

    def __getitem__(self, index):
        tensor_task = self.tensor_tasks[index]
        return tensor_task['images'], tensor_task['labels']

    def __len__(self):
        return len(self.tensor_tasks)

    def _read_json(self):
        json_file_path = os.path.join(self.data_directory, 'tasks.json')
        table = pd.read_json(json_file_path)
        filename_tasks = []
        for index in range(table.shape[0]):
            row = table.iloc[index]
            label = row['label']
            images_positive = row['images-positive']
            images_negative = row['images-negative']
            filename_tasks.append({
                "label": label,
                "images_positive": images_positive,
                "images_negative": images_negative
            })
        return filename_tasks

    def _to_tensor_tasks(self, filename_tasks):
        tensor_tasks = []
        for filename_task in filename_tasks:
            # create labels tensor
            positive_labels_tensor = torch.ones(
                filename_task['images_positive'].size(0))
            negative_labels_tensor = torch.zeros(
                filename_task['images_negative'].size(0))
            labels_tensor = torch.cat(
                [positive_labels_tensor, negative_labels_tensor])
            # create images_tensor
            positive_images_tensor = self._to_tensor(
                filename_task['images_positive'])
            negative_images_tensor = self._to_tensor(
                filename_task['images_negative'])
            images_tensor = torch.cat([positive_images_tensor, negative_images_tensor])
            tensor_tasks.append({
                "label": filename_task['label'],
                "images": images_tensor,
                "labels": labels_tensor
            })

        return tensor_tasks

    def _to_tensor(self, image_names, transforms):
        images_tensor = []
        for image_name in image_names:
            image_path = os.path.join(self.data_directory, 'images', image_name)
            img = Image.open(image_path)
            tensor = transforms(img)
            images_tensor.append(tensor)
        return images_tensor

def get_tasksets():
    train_tasks = TensorTask('/home/luis/sdacathon/data/meta-train/pretrain')
    validation_tasks = TensorTask(
        '/home/luis/sdacathon/data/meta-train/validation')
    test_adaptation_tasks = TensorTask(
        '/home/luis/sdacathon/data/meta-test/train')
    test_evaluation_tasks = TensorTask(
        '/home/luis/sdacathon/data/meta-test/test')
    return BenchmarkTasksets(train_tasks, validation_tasks, test_adaptation_tasks, test_evaluation_tasks)


tasksets = get_tasksets()
print(tasksets)
