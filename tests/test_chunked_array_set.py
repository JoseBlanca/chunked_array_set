import tempfile

from chunked_array_set import ChunkedArraySet

from .test_utils import generate_chunks, check_chunks_are_equal


def test_write_chunked_array_set():
    chunks = generate_chunks(num_chunks=2)
    with tempfile.TemporaryDirectory() as dir:
        disk_array_set = ChunkedArraySet(dir=dir, chunks=chunks)
        mem_array_set = ChunkedArraySet(dir=dir)
        assert len(list(disk_array_set.chunks)) == 2
        check_chunks_are_equal(mem_array_set.chunks, chunks)
        check_chunks_are_equal(disk_array_set.chunks, mem_array_set.chunks)

        new_chunks = generate_chunks(num_chunks=3)
        chunks.extend(new_chunks)
        mem_array_set.extend_chunks(new_chunks)
        disk_array_set.extend_chunks(new_chunks)
