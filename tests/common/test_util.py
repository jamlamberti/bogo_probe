import pytest

from common import util


def test_util():
    import os
    import errno
    util.make_dir('tmp')
    assert os.path.exists('tmp')
    util.make_dir('tmp')
    os.rmdir('tmp')

    # Should fail?
    with pytest.raises(OSError) as err:
        util.make_dir('/storage')
        assert err.errno == errno.EACCES
