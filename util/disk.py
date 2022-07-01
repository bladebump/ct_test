import datetime
import gzip
import io

from diskcache import FanoutCache, Disk
from diskcache.core import MODE_BINARY


class GzipDisk(Disk):

    def store(self, value, read, key=None):
        if type(value) is bytes:
            if read:
                value = value.read()
                read = False

            str_io = io.BytesIO()
            gz_file = gzip.GzipFile(mode='wb', compresslevel=1, fileobj=str_io)

            for offset in range(0, len(value), 2 ** 30):
                gz_file.write(value[offset:offset + 2 ** 30])
            gz_file.close()

            value = str_io.getvalue()
        return super().store(value, read)

    def fetch(self, mode, filename, value, read):
        value = super().fetch(mode, filename, value, read)
        if mode == MODE_BINARY:
            str_io = io.BytesIO()
            gz_file = gzip.GzipFile(mode='rb', fileobj=str_io)
            read_csio = io.BytesIO()

            while True:
                uncompressed_data = gz_file.read(2 ** 30)
                if uncompressed_data:
                    read_csio.write(uncompressed_data)
                else:
                    break
            value = read_csio.getvalue()
        return value


def get_cache(scope_str):
    return FanoutCache(r'D:\ct_cache' + scope_str, disk=GzipDisk, shards=64, timeout=1, size_limit=3e11)


cache = get_cache("raw")


@cache.memoize(typed=True, expire=1, tag='fib')
def ftest(n: int):
    if n == 1 or n == 2:
        return n
    else:
        return ftest(n - 1) + ftest(n - 2)


if __name__ == "__main__":
    pass
