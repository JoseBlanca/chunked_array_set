import tempfile
import functools

import numpy

from chunked_array_set import ChunkedArraySet, concatenate_chunks
from .test_utils import (
    generate_chunks,
    check_chunks_are_equal,
    check_arrays_in_two_dicts_are_equal,
)


def test_write_chunked_array_set():
    chunks = generate_chunks(num_chunks=2)
    with tempfile.TemporaryDirectory() as dir:
        disk_array_set = ChunkedArraySet(dir=dir, chunks=chunks)
        mem_array_set = ChunkedArraySet(chunks=chunks)
        assert len(list(disk_array_set.get_chunks())) == 2
        check_chunks_are_equal(mem_array_set.get_chunks(), chunks)
        check_chunks_are_equal(disk_array_set.get_chunks(), mem_array_set.get_chunks())

        new_chunks = generate_chunks(num_chunks=3)
        chunks.extend(new_chunks)
        mem_array_set.extend_chunks(new_chunks)
        disk_array_set.extend_chunks(new_chunks)


def test_chunks_different_num_rows():
    chunks = generate_chunks(num_chunks=2)
    chunks = list(chunks)
    expected_arrays = concatenate_chunks(chunks)
    num_rows_expected = {3: [3, 3, 3, 1], 5: [5, 5], 20: [10]}
    for num_rows, expected_rows_in_chunks in num_rows_expected.items():
        array_set = ChunkedArraySet(chunks=chunks, desired_num_rows_per_chunk=num_rows)
        assert array_set._get_num_rows_per_chunk() == expected_rows_in_chunks
        arrays = concatenate_chunks(array_set._chunks)
        check_chunks_are_equal([arrays], [expected_arrays])


def test_pipeline():
    chunks = generate_chunks(num_chunks=2)
    array_set = ChunkedArraySet(chunks=chunks)

    map_functs = [lambda chunk: {id: array * 2 for id, array in chunk.items()}]

    def sum_columns(current_sum, chunk):
        for id in chunk.keys():
            array = chunk[id]
            this_sum = array.sum(axis=0)
            if current_sum is None:
                current_sum = {}
            accumulated_sum_for_this_array = current_sum.get(id)
            if accumulated_sum_for_this_array is None:
                current_sum[id] = this_sum
            else:
                current_sum[id] = current_sum[id] + this_sum
        return current_sum

    result_chunks = map(map_functs[0], chunks)
    expected_result = functools.reduce(sum_columns, result_chunks, None)
    result = array_set.run_pipeline(
        map_functs=map_functs, reduce_funct=sum_columns, reduce_initialializer=None
    )

    check_arrays_in_two_dicts_are_equal(expected_result, result)

    result_chunks = list(map(map_functs[0], chunks))
    processed_chunks = array_set.run_pipeline(map_functs=map_functs)
    processed_array_set = ChunkedArraySet(chunks=processed_chunks)

    check_chunks_are_equal(processed_array_set.chunks, result_chunks)


def _check_arrays_equal_to_stacked_chunks(arrays, chunks):
    for id, array1 in arrays.items():
        array2 = numpy.vstack([chunk[id] for chunk in chunks])
        assert numpy.allclose(array1, array2)


def test_pipeline():
    chunks = generate_chunks(num_chunks=2)
    array_set = ChunkedArraySet(chunks=chunks)
    arrays = array_set.load_arrays_in_memory()
    _check_arrays_equal_to_stacked_chunks(arrays, chunks)

    with tempfile.TemporaryDirectory() as dir:
        array_set = ChunkedArraySet(chunks=chunks, dir=dir)
        arrays = array_set.load_arrays_in_memory()
        _check_arrays_equal_to_stacked_chunks(arrays, chunks)


def test_metadata():
    chunks = generate_chunks(num_chunks=2)
    with tempfile.TemporaryDirectory() as dir:
        array_set = ChunkedArraySet(chunks=chunks, dir=dir)
        array_set.metadata = "hola"
        assert array_set.metadata == "hola"


def test_num_rows():
    chunks = generate_chunks(num_chunks=2)
    with tempfile.TemporaryDirectory() as dir:
        array_set = ChunkedArraySet(chunks=chunks, dir=dir)
        assert array_set.num_rows == 10

    array_set = ChunkedArraySet(chunks=chunks)
    assert array_set.num_rows == 10


def test_get_chunks():
    chunks = generate_chunks(num_chunks=2)
    with tempfile.TemporaryDirectory() as dir:
        array_set = ChunkedArraySet(chunks=chunks, dir=dir)
        chunks = array_set.get_chunks(desired_arrays=[1])
        assert not set(
            array_id for chunk in chunks for array_id in list(chunk.keys())
        ).difference(
            {
                1,
            }
        )
