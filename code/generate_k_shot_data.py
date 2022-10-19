"""This script samples K examples randomly without replacement from the original data."""

import argparse
import os
import numpy as np
import json


def get_label(line):
    return line['relation']


def load_datasets(data_dir):
    dataset = {}
    splits = ["train", "dev", "test"]
    for split in splits:
        filename = os.path.join(data_dir, f"{split}.json")
        with open(filename, "r") as f:
            data = json.load(f)
        dataset[split] = data
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=16,
                        help="Training examples for each class.")
    parser.add_argument("--seed", type=int, nargs="+",
                        default=[100, 13, 21, 42, 87],
                        help="Random seeds")

    parser.add_argument("--data_dir", type=str, default="../data/re-tacred", help="Path to original data")
    parser.add_argument("--output_dir", type=str, default="../data/re-tacred", help="Output path")
    parser.add_argument("--mode", type=str, default='k-shot', choices=['k-shot', 'k-shot-10x'],
                        help="k-shot or k-shot-10x (10x dev set)")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.mode)

    k = args.k
    print("K =", k)
    dataset = load_datasets(args.data_dir)

    for seed in args.seed:
        print("Seed = %d" % seed)

        # Set random seed
        np.random.seed(seed)

        # Shuffle the training set
        train_lines = dataset['train']
        np.random.shuffle(train_lines)
        val_lines = dataset['dev']
        np.random.shuffle(val_lines)

        # Set up dir
        setting_dir = os.path.join(args.output_dir, f"{k}-{seed}")
        os.makedirs(setting_dir, exist_ok=True)

        # Write test splits

        with open(os.path.join(os.path.join(setting_dir, 'test.json')), "w") as f:
            json.dump(dataset['test'], f)

        # Get label list for balanced sampling
        label_list = {}
        for line in train_lines:
            label = get_label(line)
            if label not in label_list:
                label_list[label] = [line]
            else:
                label_list[label].append(line)

        new_train = []
        for label in label_list:
            for line in label_list[label][:k]:
                new_train.append(line)
        with open(os.path.join(setting_dir, 'train.json'), 'w') as f:
            json.dump(new_train, f)

        va_label_list = {}
        for line in val_lines:
            label = get_label(line)
            if label not in va_label_list:
                va_label_list[label] = [line]
            else:
                va_label_list[label].append(line)

        new_dev = []
        for label in va_label_list:
            dev_rate = 10 if '10x' in args.mode else 1
            for line in va_label_list[label][:k * dev_rate]:
                new_dev.append(line)
        with open(os.path.join(setting_dir, 'dev.json'), 'w') as f:
            json.dump(new_dev, f)


if __name__ == "__main__":
    main()
