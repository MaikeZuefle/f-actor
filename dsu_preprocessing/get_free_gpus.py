# get_free_gpus.py (v2 - with error handling)
import argparse
import sys

import pynvml


def get_best_gpus(num_gpus):
    """
    Returns a comma-separated string of the best GPU device IDs.
    Returns None on failure.
    """
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        if device_count < num_gpus:
            print(
                f"ERROR: Requested {num_gpus} GPUs, but only {device_count} are available.",
                file=sys.stderr,
            )
            return None

        gpu_memory = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory.append((i, mem_info.free))

        # Sort GPUs by free memory in descending order
        gpu_memory.sort(key=lambda x: x[1], reverse=True)

        # Get the indices of the top N GPUs
        best_gpus = [str(gpu[0]) for gpu in gpu_memory[:num_gpus]]

        return ",".join(best_gpus)

    except pynvml.NVMLError as error:
        print(f"ERROR: Failed to query NVIDIA GPUs: {error}", file=sys.stderr)
        return None

    finally:
        # This will run whether there was an error or not
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass  # Ignore shutdown errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find GPUs with the most free memory.")
    parser.add_argument(
        "--num_gpus", type=int, required=True, help="Number of GPUs to select."
    )
    args = parser.parse_args()

    best_gpu_ids = get_best_gpus(args.num_gpus)

    if best_gpu_ids:
        print(best_gpu_ids)
    else:
        # Exit with a non-zero code to indicate failure
        sys.exit(1)
