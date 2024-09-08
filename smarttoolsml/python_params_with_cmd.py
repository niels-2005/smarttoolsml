import argparse

# Example usage: python python_params_with_cmd.py -p1 10 -p2 10


# simple eval function
def eval(p1, p2):
    output_metric = p1**2 + p2**2
    return output_metric


# main function
def main(inp1, inp2):
    metric = eval(inp1, inp2)
    return metric


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    # add arguments to cmd commando
    args.add_argument("--param1", "-p1", type=int, default=5)
    args.add_argument("--param2", "-p2", type=int, default=10)
    parsed_args = args.parse_args()

    # parsed_args.param1
    main(parsed_args.param1, parsed_args.param2)
