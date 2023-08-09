from functools import partial

import numpy
import pandas


def create_normal_array(shape, loc=0.0, scale=1.0):
    return numpy.random.default_rng().normal(loc=loc, scale=scale, size=shape)


def create_numpy_arrays(array_ids, array_generator):
    return {array_id: next(array_generator) for array_id in array_ids}


def generic_chunk_generator(num_arrays, array_generator):
    return {id_: array_generator() for id_ in range(num_arrays)}


DEFAULT_CHUNK_GENERATOR = partial(
    generic_chunk_generator,
    num_arrays=3,
    array_generator=partial(create_normal_array, shape=(5, 20)),
)


def generate_chunks(num_chunks=2, chunk_generator=DEFAULT_CHUNK_GENERATOR):
    dataset = []
    for _ in range(num_chunks):
        dataset.append(chunk_generator())
    return dataset


def check_arrays_in_two_dicts_are_equal(arrays1: dict, arrays2: dict):
    assert not set(arrays1.keys()).difference(arrays2.keys())

    for id in arrays1.keys():
        array1 = arrays1[id]
        array2 = arrays2[id]
        assert type(array1) == type(array2)

        if isinstance(array1, numpy.ndarray):
            if numpy.issubdtype(array1.dtype, float):
                assert numpy.allclose(array1, array2)
            else:
                assert numpy.allequal(array1, array2)
        elif isinstance(array1, pandas.DataFrame):
            assert array1.equals(array2)


def check_chunks_are_equal(chunks1, chunks2):
    for arrays1, arrays2 in zip(chunks1, chunks2):
        check_arrays_in_two_dicts_are_equal(arrays1, arrays2)
