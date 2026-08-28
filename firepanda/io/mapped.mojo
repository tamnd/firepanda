"""Handing a file to the reader without copying it first.

`read_csv` used to open the file, ask for its bytes, and hand the resulting
`List` to the parser. That is one full copy of the file out of the page cache
and into the process, and on a 375 MB file it was 394 ms, which by the time the
string columns stopped being copied element by element was the single largest
thing a read did. It also doubled the peak footprint of a read, because the copy
and the columns built from it are alive at the same time.

`mmap` removes both. The kernel already has the bytes, and a mapping points at
them where they are. Nothing is copied, the pages arrive on demand as the parser
touches them, and because the parser touches its blocks on every core at once,
the faults are taken in parallel rather than serialised behind one `read` call.

The mapping is read only and private, so a write through it would fault rather
than reach the file, and the file changing underneath a live mapping is the
caller's problem in exactly the way it already was with a copy.

The file is opened with the ordinary `open`, and only the descriptor underneath
it goes to libc. Calling `open` through the FFI directly does not compile,
because the standard library already declares that symbol with a different
signature, and going through the file handle is better anyway: a path that does
not exist raises the error the rest of the library raises for it rather than an
errno this file would have to translate.

Three libc calls and three constants, and all three constants are the same
number on Linux and on macOS, which is every platform the CI matrix builds:

    SEEK_END      2    seek relative to the end, which is how the size is found
    PROT_READ     1    the mapping may be read
    MAP_PRIVATE   2    writes, if there were any, would not reach the file

`MAP_POPULATE` is deliberately not used. It is Linux only, it would make the
constants stop being portable, and it moves the fault cost back in front of the
parse where it is serial again rather than leaving it spread across the workers.

Every failure here raises, and every caller is expected to fall back to reading
the bytes the old way. `map_file` is the wrapper that turns the failure into a
value, because that is the shape a caller with a fallback wants.
"""

from std.collections.span import Span
from std.ffi import external_call

comptime SEEK_END = Int32(2)
"""Seek relative to the end of the file."""

comptime PROT_READ = Int32(1)
"""The mapping may be read."""

comptime MAP_PRIVATE = Int32(2)
"""Changes to the mapping are private to this process."""

comptime MAP_FAILED = -1
"""What `mmap` returns instead of an address, as an integer."""


struct MappedFile(Movable, Sized):
    """A whole file mapped into memory, read only.

    The mapping is unmapped when the value dies, so the `Span` handed out by
    `bytes` borrows from it and cannot outlive it. That is the whole safety
    story, and the origin system enforces it.
    """

    var _addr: Pointer[UInt8, ImmUntrackedOrigin]
    """The base of the mapping.

    Untracked because a struct field cannot carry a borrow of something outside
    the struct, and there is nothing outside the struct to borrow from: the
    mapping is owned here and released in `__deinit__`.
    """

    var _size: Int
    """The size of the mapping, which the file had when it was opened."""

    def __init__(out self, path: String) raises:
        """Maps a file.

        Args:
            path: The file to map.

        Raises:
            Error: If the file cannot be opened, is empty, or cannot be mapped.
                The caller is expected to fall back to copying the bytes.
        """
        var handle = open(path, "r")
        var fd = Int32(handle._get_raw_fd())
        var size = external_call["lseek", Int64](fd, Int64(0), SEEK_END)
        if size <= 0:
            # An empty file has nothing to map, and `mmap` rejects a length of
            # zero rather than handing back an empty mapping.
            handle.close()
            raise Error(String("nothing to map in '", path, "'"))
        var addr = external_call["mmap", Pointer[UInt8, ImmUntrackedOrigin]](
            Int(0), Int(size), PROT_READ, MAP_PRIVATE, fd, Int64(0)
        )
        # The descriptor is not needed once the mapping exists. The mapping
        # holds its own reference to the file.
        handle.close()
        if Int(addr) == MAP_FAILED:
            raise Error(String("cannot map '", path, "'"))
        self._addr = addr
        self._size = Int(size)

    def __deinit__(deinit self):
        """Unmaps the file."""
        _ = external_call["munmap", Int32](self._addr, self._size)

    def __len__(self) -> Int:
        """Returns the size of the file in bytes.

        Returns:
            The size the file had when it was mapped.
        """
        return self._size

    def bytes(imm self) -> Span[UInt8, origin_of(self)]:
        """Returns the file's bytes.

        Returns:
            A span over the whole mapping, valid for as long as this value is.
        """
        return Span[UInt8, origin_of(self)](
            unsafe_ptr=self._addr.unsafe_origin_cast[origin_of(self)](),
            length=self._size,
        )


def map_file(path: String) -> Optional[MappedFile]:
    """Maps a file, or reports that it could not be mapped.

    A caller with a fallback wants the failure as a value rather than as an
    exception, because catching around the whole read would also catch the
    parse and quietly redo it.

    Args:
        path: The file to map.

    Returns:
        The mapping, or nothing if the file could not be mapped for any reason.
    """
    try:
        return Optional(MappedFile(path))
    except:
        return None
