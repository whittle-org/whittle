from argparse import ArgumentParser

from litgpt.eval.evaluate import convert_and_evaluate as evaluate_fn


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")

    args, _ = parser.parse_known_args()

    checkpoint_dir = args.checkpoint_dir
    out_dir = args.output_dir
    tasks = ["hellaswag"]

    evaluate_fn(checkpoint_dir=checkpoint_dir, tasks=",".join(tasks), out_dir=out_dir)
