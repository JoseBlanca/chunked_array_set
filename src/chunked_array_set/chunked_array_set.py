from collections.abc import Iterator
from typing import Hashable, Callable
from pathlib import Path
import pickle
import functools
import json
from collections import defaultdict

import numpy
import pandas

Array = tuple[numpy.ndarray, pandas.DataFrame]
ARRAY_FILE_EXTENSIONS = {"DataFrame": ".parquet", "ndarray": ".npy"}
USER_METADATA_FNAME = "user_metadata.json"


def _get_metadata_chunk_path(chunk_dir):
    return chunk_dir / "metadata.pickle"


def _write_chunk(chunk_dir, chunk):
    arrays_metadata = []
    num_rows = None
    for array_id, array in chunk.items():
        type_name = type(array).__name__
        if type_name not in ("DataFrame", "ndarray"):
            raise ValueError(f"Don't know how to store type: {type(array)}")

        array_path = chunk_dir / f"id:{array_id}{ARRAY_FILE_EXTENSIONS[type_name]}"

        if type_name == "DataFrame":
            array.to_parquet(array_path)
            save_method = "parquet"
        elif type_name == "ndarray":
            numpy.save(array_path, array)
            save_method = "npy"

        this_array_num_rows = array.shape[0]
        if num_rows is None:
            num_rows = this_array_num_rows
        else:
            if num_rows != this_array_num_rows:
                raise ValueError("Arrays have different number of rows")

        arrays_metadata.append(
            {
                "type_name": type_name,
                "array_path": array_path,
                "save_method": save_method,
                "id": array_id,
            }
        )

    metadata = {"arrays_metadata": arrays_metadata}
    metadata_path = _get_metadata_chunk_path(chunk_dir)
    with metadata_path.open("wb") as fhand:
        pickle.dump(metadata, fhand)

    return {"num_rows": num_rows}


def _write_chunks(chunks, dir):
    dir = Path(dir)
    if not dir.is_dir():
        raise ValueError(f"dir should be a dir: {dir}")

    metadata_path = _get_metadata_chunk_path(dir)
    if metadata_path.exists():
        with metadata_path.open("rb") as fhand:
            metadata = pickle.load(fhand)
    else:
        metadata = {"chunks_metadata": []}

    chunks_metadata = metadata["chunks_metadata"]
    num_previous_chunks = len(chunks_metadata)
    for idx, chunk in enumerate(chunks):
        id = idx + num_previous_chunks
        chunk_dir = dir / f"dataset_chunk:{id:08}"
        chunk_dir.mkdir()
        res = _write_chunk(chunk_dir, chunk)
        num_rows = res["num_rows"]
        chunks_metadata.append({"id": id, "dir": chunk_dir, "num_rows": num_rows})

    with metadata_path.open("wb") as fhand:
        pickle.dump(metadata, fhand)


def _get_array_set_metadata(chunk_dir):
    metadata_path = _get_metadata_chunk_path(chunk_dir)
    with metadata_path.open("rb") as fhand:
        metadata = pickle.load(fhand)
    return metadata


def _load_chunk(chunk_dir, desired_arrays):
    metadata = _get_array_set_metadata(chunk_dir)

    arrays = {}
    for array_metadata in metadata["arrays_metadata"]:
        id = array_metadata["id"]
        if desired_arrays and id not in desired_arrays:
            continue
        if array_metadata["save_method"] == "parquet":
            array = pandas.read_parquet(array_metadata["array_path"])
        elif array_metadata["save_method"] == "npy":
            array = numpy.load(array_metadata["array_path"])
        arrays[id] = array
    return arrays


def _load_chunks(dir, desired_arrays: list | None):
    metadata_path = _get_metadata_chunk_path(dir)
    with metadata_path.open("rb") as fhand:
        metadata = pickle.load(fhand)
    chunks_metadata = metadata["chunks_metadata"]

    for chunk_metadata in chunks_metadata:
        chunk_dir = chunk_metadata["dir"]
        yield _load_chunk(chunk_dir, desired_arrays=desired_arrays)


def _pandas_empty_like(array, shape):
    empty_array = {}
    for col, data in array.items():
        empty_array[col] = numpy.empty(shape=(shape[0],), dtype=data.dtype)
    empty_array = pandas.DataFrame(empty_array)
    return empty_array


class ChunkedArraySet:
    def __init__(
        self,
        chunks: Iterator[dict[Hashable, Array]] | None = None,
        dir: Path | None = None,
        metadata=None,
        desired_num_rows_per_chunk: int | None = None,
    ):
        if dir:
            dir = Path(dir)
            dir.mkdir(exist_ok=True)
            self._in_memory = False
        else:
            self._in_memory = True

        self._dir = dir

        self._desired_num_rows_per_chunk = desired_num_rows_per_chunk

        self._metadata = None
        if metadata is not None:
            self.metadata = metadata

        self._chunks = []
        if chunks:
            self.extend_chunks(chunks)

    def _set_metadata(self, metadata):
        if self._in_memory:
            self._metadata = metadata
        else:
            with (self._dir / USER_METADATA_FNAME).open("wt") as fhand:
                json.dump(metadata, fhand)

    def _get_metadata(self):
        if self._in_memory:
            metadata = self._metadata
        else:
            try:
                with (self._dir / USER_METADATA_FNAME).open("rt") as fhand:
                    metadata = json.load(fhand)
            except FileNotFoundError:
                metadata = None
        return metadata

    metadata = property(_get_metadata, _set_metadata)

    def get_chunks(self, desired_arrays: list | None = None):
        if self._chunks:
            return iter(self._chunks)
        if self._dir:
            return _load_chunks(self._dir, desired_arrays=desired_arrays)

    def extend_chunks(self, chunks: Iterator[dict[Hashable, Array]]):
        if self._desired_num_rows_per_chunk:
            chunks = _normalize_num_rows_in_chunk(
                chunks, self._desired_num_rows_per_chunk
            )

        if self._in_memory:
            self._chunks.extend(chunks)
        else:
            _write_chunks(chunks, self._dir)

    def run_pipeline(
        self,
        desired_arrays_to_load_in_chunk: list | None = None,
        map_functs: list[Callable] | None = None,
        reduce_funct: Callable | None = None,
        reduce_initialializer=None,
    ):
        if map_functs is None:
            map_functs = []

        def funct(item):
            processed_item = item
            for one_funct in map_functs:
                processed_item = one_funct(processed_item)
            return processed_item

        processed_chunks = map(
            funct, self.get_chunks(desired_arrays=desired_arrays_to_load_in_chunk)
        )
        result = processed_chunks

        if reduce_funct:
            reduced_result = functools.reduce(
                reduce_funct, processed_chunks, reduce_initialializer
            )
            result = reduced_result

        return result

    def _get_num_rows_per_chunk(self):
        if self._in_memory:
            num_rows = []
            for chunk in self.get_chunks():
                try:
                    one_array = next(iter(chunk.values()))
                except StopIteration:
                    raise RuntimeError("Chunk with no arrays")
                if isinstance(one_array, (pandas.DataFrame, numpy.ndarray)):
                    num_rows_this_chunk = one_array.shape[0]
                num_rows.append(num_rows_this_chunk)
        else:
            metadata = _get_array_set_metadata(self._dir)
            chunks_metadata = metadata["chunks_metadata"]
            num_rows = [
                chunk_metadata["num_rows"] for chunk_metadata in chunks_metadata
            ]
        return num_rows

    @property
    def num_rows(self):
        return sum(self._get_num_rows_per_chunk())

    def load_arrays_in_memory(self):
        return _collect_chunks_in_memory(self.get_chunks(), self.num_rows)


def _get_num_rows_in_chunk(buffered_chunk):
    if not buffered_chunk:
        return 0
    else:
        return list(buffered_chunk.values())[0].shape[0]


def _fill_buffer(buffered_chunk, chunks, desired_num_rows):
    num_rows_in_buffer = _get_num_rows_in_chunk(buffered_chunk)
    if num_rows_in_buffer >= desired_num_rows:
        return buffered_chunk, False

    chunks_to_concat = []
    if num_rows_in_buffer:
        chunks_to_concat.append(buffered_chunk)

    total_num_rows = num_rows_in_buffer
    no_chunks_remaining = True
    for chunk in chunks:
        total_num_rows += _get_num_rows_in_chunk(chunk)
        chunks_to_concat.append(chunk)
        if total_num_rows >= desired_num_rows:
            no_chunks_remaining = False
            break

    if not chunks_to_concat:
        buffered_chunk = None
    elif len(chunks_to_concat) > 1:
        buffered_chunk = concatenate_chunks(chunks_to_concat)
    else:
        buffered_chunk = chunks_to_concat[0]
    return buffered_chunk, no_chunks_remaining


def _yield_chunks_from_buffer(buffered_chunk, desired_num_rows):
    num_rows_in_buffer = _get_num_rows_in_chunk(buffered_chunk)
    if num_rows_in_buffer == desired_num_rows:
        chunks_to_yield = [buffered_chunk]
        buffered_chunk = None
        return buffered_chunk, chunks_to_yield

    start_row = 0
    chunks_to_yield = []
    end_row = None
    while True:
        previous_end_row = end_row
        end_row = start_row + desired_num_rows
        if end_row <= num_rows_in_buffer:
            chunks_to_yield.append(
                get_rows_from_chunk(buffered_chunk, start_row, end_row)
            )
        else:
            end_row = previous_end_row
            break
        start_row = end_row

    remainder = get_rows_from_chunk(buffered_chunk, end_row, None)
    buffered_chunk = remainder
    return buffered_chunk, chunks_to_yield


def _normalize_num_rows_in_chunk(chunks, desired_num_rows):
    buffered_chunk = None
    chunks = iter(chunks)

    while True:
        # fill buffer with equal or more than desired
        buffered_chunk, no_chunks_remaining = _fill_buffer(
            buffered_chunk, chunks, desired_num_rows
        )
        # yield chunks until buffer less than desired
        num_rows_in_buffer = _get_num_rows_in_chunk(buffered_chunk)
        if not num_rows_in_buffer:
            break
        buffered_chunk, chunks_to_yield = _yield_chunks_from_buffer(
            buffered_chunk, desired_num_rows
        )
        for chunk in chunks_to_yield:
            yield chunk

        if no_chunks_remaining:
            yield buffered_chunk
            break


def _pandas_empty_like(array, shape):
    empty_array = {}
    for col, data in array.items():
        empty_array[col] = numpy.empty(shape=(shape[0],), dtype=data.dtype)
    empty_array = pandas.DataFrame(empty_array)
    return empty_array


def create_empty_array_like(sample_array, num_rows):
    shape = list(sample_array.shape)
    shape[0] = num_rows

    if isinstance(sample_array, numpy.ndarray):
        array = numpy.empty_like(sample_array, shape=shape)
    elif isinstance(sample_array, pandas.DataFrame):
        array = _pandas_empty_like(sample_array, shape)
    return array


def set_array_chunk(complete_array, array_chunk, row_start, row_end):
    if isinstance(array_chunk, numpy.ndarray):
        complete_array[row_start:row_end, :] = array_chunk
    elif isinstance(array_chunk, pandas.DataFrame):
        for col in complete_array.columns:
            complete_array[col][row_start:row_end] = array_chunk[col]


def get_rows_from_chunk(chunk, row_start, row_end):
    small_chunk = {}
    for array_id, array in chunk.items():
        if isinstance(array, numpy.ndarray):
            small_array = array[row_start:row_end, :]
        elif isinstance(array, pandas.DataFrame):
            small_array = array.iloc[row_start:row_end, :]
        small_chunk[array_id] = small_array
    return small_chunk


def filter_array_chunk_rows(array_chunk, rows_to_keep):
    if isinstance(array_chunk, numpy.ndarray):
        array_chunk = array_chunk[rows_to_keep, ...]
    elif isinstance(array_chunk, pandas.DataFrame):
        array_chunk = array_chunk.loc[rows_to_keep, :]
    return array_chunk


def concatenate_chunks(array_chunks: list[dict]):
    if len(array_chunks) == 1:
        return array_chunks[0]

    arrays_to_concatenate = defaultdict(list)
    for chunk in array_chunks:
        for array_id, array in chunk.items():
            arrays_to_concatenate[array_id].append(array)

    num_arrays = [len(arrays) for arrays in arrays_to_concatenate.values()]
    if not all([num_arrays[0] == len_ for len_ in num_arrays]):
        raise ValueError("Nota all chunks have the same arrays")

    concatenated_chunk = {}
    for array_id, arrays in arrays_to_concatenate.items():
        if isinstance(arrays[0], numpy.ndarray):
            array = numpy.vstack(arrays)
        elif isinstance(arrays[0], pandas.DataFrame):
            array = pandas.concat(arrays, axis=0)
        concatenated_chunk[array_id] = array
    return concatenated_chunk


def _collect_chunks_in_memory(chunks, num_rows):
    arrays = {}
    row_start = 0
    for chunk in chunks:
        row_end = None
        for id, array_chunk in chunk.items():
            if id not in arrays:
                shape = list(array_chunk.shape)
                shape[0] = num_rows
                if isinstance(array_chunk, numpy.ndarray):
                    array = numpy.empty_like(array_chunk, shape=shape)
                elif isinstance(array_chunk, pandas.DataFrame):
                    array = _pandas_empty_like(array_chunk, shape)
                arrays[id] = array
            else:
                array = arrays[id]
            if row_end is None:
                row_end = row_start + array_chunk.shape[0]

            if isinstance(array_chunk, numpy.ndarray):
                array[row_start:row_end, :] = array_chunk
            elif isinstance(array_chunk, pandas.DataFrame):
                for col in array.columns:
                    array[col][row_start:row_end] = array_chunk[col]
        row_start = row_end
    return arrays
