import argparse
import json


def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Check index against the_error_is=1.')
    parser.add_argument('index', type=int, help='Integer index to check.')
    args = parser.parse_args()

    the_error_is = 1004

    # Load existing metadata if available, otherwise use an empty list
    try:
        with open('metadata.json', 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = []

    # Check the argument against the error condition
    if args.index == the_error_is:
        print('Ok you got the error, please mark the task as done')
        metadata.append(1)
    else:
        print('No error')
        metadata.append(0)

    # Save the updated metadata
    with open('metadata.json', 'w') as f:
        json.dump(metadata, f)


if __name__ == '__main__':
    main()
