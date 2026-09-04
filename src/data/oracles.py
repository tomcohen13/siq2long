
from pathlib import Path
from typing import Any, Hashable, Mapping

def load_oracles(path_to_trims: str) -> dict[Hashable, dict[Hashable, Any]]:
    """
    Load SIQ2 oracles per video, computed as the absolute time range, in seconds,
    of the SIQ2 trimmed video within the full one.

    Args:
        path_to_trims: path to the original SocialIQ-2.0 trims.json file
        dataset: alternative to path, 
    """
    import pandas as pd
    trims_df = pd.read_json(path_to_trims, orient="index")
    trims_df.rename(columns={0: "start"}, inplace=True)
    trims_df["end"] = round(trims_df["start"] + 60, 5)
    trims_df["oracle"] = trims_df[["start", "end"]].apply(tuple, axis=1)
    return trims_df.to_dict(orient="index")


def compute_segments_around_oracle(
    oracle: tuple[float, float],
    full_duration: float,
    chunk_size: int = 60,  # seconds
    buffer_size: int = 10, # seconds
) -> tuple[list, int]:
    """
    Compute non-overlapping time ranges to segment a video by, around oracle.

    Args:
        oracle: a list of start time and end time of the oracle
        full_duration: full duration (in seconds) of the video
        chunk_size: desired time range for each chunk
        buffer_size: if the outermost chunks end up being within buffer_size seconds of start or end of video,
            absorb that into chunk. Never applied when the absorbing chunk would be the oracle --
            the sliver is dropped instead, so the oracle's bounds always match the SIQ2 trim.

    Returns:
        chunks (list[list[float]])
        oracle_idx (int)

    Example:
        >>> compute_segments_around_oracle((127.11, 187.11), 200)
        ([[0, 67.11], [67.11, 127.11], [127.11, 187.11], [187.11, 200]], 2)
        >>> compute_segments_around_oracle((127.11, 187.11), 190)  # 2.89s tail, dropped
        ([[0, 67.11], [67.11, 127.11], [127.11, 187.11]], 2)
    """

    chunks = []
    oracle_start, oracle_end = oracle
    chunk_start, chunk_end = oracle_start - chunk_size, oracle_start
    while chunk_start >= 0:
        chunks.append([chunk_start, chunk_end])
        chunk_end = chunk_start
        chunk_start -= chunk_size
    # If the leftmost window starts within BUFFER_TIME of 0, absorb the sliver into the first
    # chunk by extending its start to 0 (instead of adding a tiny [0, chunk_end] segment).
    # After the loop, chunk_start is < 0 and chunk_end is the start time of that leftmost window.
    if 0 <= chunk_end <= buffer_size:
        if chunks:
            chunks[-1][0] = 0
    else:
        chunks.append([0, chunk_end])
    
    chunks = chunks[::-1]
    chunks.append([oracle_start, oracle_end])
    oracle_idx = len(chunks) - 1

    chunk_start, chunk_end = oracle_end, oracle_end + chunk_size
    while chunk_end <= full_duration:
        chunks.append([chunk_start, chunk_end])
        chunk_start = chunk_end
        chunk_end += chunk_size

    # apply buffer to remainder, unless last chunk is oracle.
    remainder = full_duration - chunk_start
    if remainder > buffer_size:
        chunks.append([chunk_start, full_duration])
    elif len(chunks) > oracle_idx + 1:
        chunks[-1][1] = full_duration

    return chunks, oracle_idx
