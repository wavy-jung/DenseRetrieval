import os
from datasets import load_dataset
import argparse

def main(args):
    base_path = "../dataset/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    if args.data_name == "tevatron":
        tevatron = load_dataset("Tevatron/msmarco-passage")
        tevatron.save_to_disk(os.path.join(base_path, "/tevatron"))

    elif args.data_name == "squad2":
        squad2 = load_dataset("squad_v2")
        squad2.save_to_disk(os.path.join(base_path, "squad_v2"))

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="tevatron")
    args = parser.parse_args()
    main(args)