import numpy as np
import contextlib

FLOHASH = 1212500304

# ----------------------------------------------------------------------------
# Read and Write .flo format
# ----------------------------------------------------------------------------
@contextlib.contextmanager
def openflow(filename, mode='r', shape=None):
    """Open a .flo file for reading or writing

    This handles opening and closing of the flo file, as well as checking
    headers. The object returned is a numpy memmap, which has the read/write
    properties passed to the function. The open function works as a context
    manager:

    with openflow('path/to/file.flo', 'w+', shape=(M,N)) as flow:
        flow[:] = np.dstack((fx,fy))

    Args:
        filename: The path to the .flo file

    Keyword Args:
        mode: The open file mode passed to np.memmap. Should be one of:
            'r', 'r+', 'w+', or 'c'. (Default: 'r')
        shape: If the file is open for writing ('w+'), shape defines the spatial
            dimensionality (M,N) of the flow fields to store. (Default: None)
    """
    write = mode == 'w+'
    size  = (3+2*shape[0]*shape[1],) if shape else shape

    # memory-map the file
    filemap = np.memmap(filename, mode=mode, shape=size, dtype=np.single)

    # context protected
    try:
        # parse the header
        header = filemap[:3].view(np.int32)
        hashval, N, M = (FLOHASH, shape[1], shape[0]) if write else header[:]
        if write:
            header[:] = hashval, N, M
        elif not hashval == FLOHASH:
            raise IOError('File format not recognized')

        # yield the flow containers
        yield filemap[3:].reshape(M,N,2).transpose(2,0,1)

    finally:
        # cleanup
        del filemap
