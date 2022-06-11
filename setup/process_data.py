import os
from datasets import load_dataset
import argparse

def save_data(dataname: str, path: str):
    if not os.path.exists(path):
        data = load_dataset(dataname)
        data.save_to_disk(path)
    else:
        print("dataset already saved to disk")


def cache_data(args):
    base_path = "./dataset/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    if args.data_name == "tevatron":
        save_data("Tevatron/msmarco-passage", os.path.join(base_path, "tevatron"))
        save_data("Tevatron/msmarco-passage-corpus", os.path.join(base_path, "tevatron-corpus"))

    elif args.data_name == "squad2" or args.data_name == "squad_v2":
        save_data("sqaud_v2", os.path.join(base_path, "squad_v2"))

    elif args.data_name == "adversarial":
        save_data("adversarial_qa", os.path.join(base_path, "adversarial_qa"))

    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="tevatron")
    args = parser.parse_args()
    cache_data(args)