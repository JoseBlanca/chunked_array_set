from collections.abc import Iterator
from typing import Hashable, Callable
from pathlib import Path
import pickle
import functools

import numpy
import pandas

Array = tuple[numpy.ndarray, pandas.DataFrame]
ARRAY_FILE_EXTENSIONS = {"DataFrame": ".parquet", "ndarray": ".npy"}


def _get_metadata_chunk_path(chunk_dir):
    return chunk_dir / "metadata.pickle"


def _write_chunk(chunk_dir, chunk):
    arrays_metadata = []
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
        _write_chunk(chunk_dir, chunk)
        chunks_metadata.append({"id": id, "dir": chunk_dir})

    with metadata_path.open("wb") as fhand:
        pickle.dump(metadata, fhand)


def _load_chunk(chunk_dir):
    metadata_path = _get_metadata_chunk_path(chunk_dir)
    with metadata_path.open("rb") as fhand:
        metadata = pickle.load(fhand)

    arrays = {}
    for array_metadata in metadata["arrays_metadata"]:
        if array_metadata["save_method"] == "parquet":
            array = pandas.read_parquet(array_metadata["array_path"])
        elif array_metadata["save_method"] == "npy":
            array = numpy.load(array_metadata["array_path"])
        arrays[array_metadata["id"]] = array
    return arrays


def _load_chunks(dir):
    metadata_path = _get_metadata_chunk_path(dir)
    with metadata_path.open("rb") as fhand:
        metadata = pickle.load(fhand)
    chunks_metadata = metadata["chunks_metadata"]

    for chunk_metadata in chunks_metadata:
        chunk_dir = chunk_metadata["dir"]
        yield _load_chunk(chunk_dir)


class ChunkedArraySet:
    def __init__(
        self,
        chunks: Iterator[dict[Hashable, Array]] | None = None,
        dir: Path | None = None,
    ):
        if dir:
            dir = Path(dir)
            self._in_memory = False
        else:
            self._in_memory = True

        self._dir = dir

        self._chunks = []
        if chunks:
            self.extend_chunks(chunks)

    @property
    def chunks(self):
        if self._chunks:
            return iter(self._chunks)
        if self._dir:
            return _load_chunks(self._dir)

    def extend_chunks(self, chunks: Iterator[dict[Hashable, Array]]):
        if self._in_memory:
            self._chunks.extend(chunks)
        else:
            _write_chunks(chunks, self._dir)

    def run_pipeline(
        self,
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

        processed_chunks = map(funct, self.chunks)
        result = processed_chunks

        if reduce_funct:
            reduced_result = functools.reduce(
                reduce_funct, processed_chunks, reduce_initialializer
            )
            result = reduced_result

        return result
