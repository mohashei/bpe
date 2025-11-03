import os
from typing import BinaryIO
import multiprocessing as mp
from collections import defaultdict

import regex as re


# Pretokenization pattern from GPT-2
_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk(args: tuple[str, bytes, int, int]) -> dict[tuple[str], int]:
    """
    Worker function: reads a byte range from a file and returns token counts.
    args: (filename, special_token, start, end)
    """
    filename, special_token, start, end = args
    with open(filename, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    tokens = defaultdict(int)
    chunks = chunk.split(special_token.decode("utf-8"))
    for chunk in chunks:
        for match in re.finditer(_PAT, chunk):
            token = match.group()
            if token:  # skip empty strings
                tokens[tuple(token)] += 1
    return tokens


def parallel_tokenize(
        filename: str,
        boundaries: list[int],
        special_token: bytes,
        num_workers: int | None = None,
) -> list[dict[tuple[str], int]]:
    if num_workers is None:
        num_workers = mp.cpu_count()

    # Prepare arguments for each chunk: (file, start, end)
    chunk_args = [(filename, special_token, start, end)
                  for start, end in zip(boundaries[:-1], boundaries[1:])]

    with mp.Pool(processes=num_workers) as pool:
        chunk_tokens = pool.map(process_chunk, chunk_args)
    
    return chunk_tokens


def pretokenize(
        filename: str, special_token: bytes, num_processes: int = 4
) -> dict[tuple[str], int]:
    with open(filename, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_token)

    chunk_tokens = parallel_tokenize(
        filename, boundaries, special_token, num_workers=num_processes
    )
    # Combine results from all chunks
    final_tokens = defaultdict(int)
    for chunk in chunk_tokens:
        for token, count in chunk.items():
            final_tokens[token] += count
    return final_tokens


if __name__ == "__main__":
    # Example usage
    pretokenized = pretokenize("TinyStories-valid.txt")
    print(len(pretokenized), list(pretokenized.items())[:10])
