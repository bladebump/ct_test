import datetime
import time
from collections import namedtuple

import numpy as np

from applog import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

IrcTuple = namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = namedtuple('XyzTuple', ['x', 'y', 'z'])


def irc2xyz(coord_irc, origin_xyz, vx_size_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1]
    origin_a = np.array(origin_xyz)
    vx_size_a = np.array(vx_size_xyz)
    coord_xyz = (direction_a @ (cri_a * vx_size_a)) + origin_a
    return XyzTuple(*coord_xyz)


def xyz2irc(coord_xyz, origin_xyz, vx_size_xyz, direction_a):
    origin_a = np.array(origin_xyz)
    vx_size_a = np.array(vx_size_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vx_size_a
    cri_a = np.round(cri_a)
    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))


def enumerate_with_estimate(iter, desc_str, start_ndx=0, print_ndx=4, backoff=None, iter_len=None):
    """
        功能上和`enumerate`上一样，只是添加了一些日志记录

        :param iter: 传递给`enumerate`的参数. 必需参数.

        :param desc_str: 描述循环正在进行的操作.类似于 `"第 4 轮训练"` 或者
            `"删除临时文件"` 或者类似的有意义的描述.

        :param start_ndx: 用来指定跳过多少轮循环，在有缓存的情况下，这样做是有意义的

            NOTE: 使用`start_ndx`去跳过循环会导致显示的时间不正常

            `start_ndx`默认值为`0`.

        :param print_ndx: 确定日志记录从哪个循环开始，这几个循环会用来计算平均时间，要确保`print_ndx` 不小于 `start_ndx`乘以`backoff`

            `print_ndx`默认为`4`.

        :param backoff: 这用于确定记录间隔.频繁的日志记录并不是很有趣，所以在第一次以后记录间隔会加倍。

            `backoff` 默认为 `2` 除非 iter_len is > 1000, 这时候默认为 `4`.

        :param iter_len: 用于计算循环的时间. 不提供会自动计算.

        :return:
    """
    if iter_len is None:
        iter_len = len(iter)

    if backoff is None:
        backoff = 2
        while backoff ** 7 < iter_len:
            backoff *= 2

    assert backoff >= 2

    while print_ndx < start_ndx * backoff:
        print_ndx *= backoff

    log.warning(f"{desc_str} ----/{iter_len},开始")
    start_ts = time.time()
    for current_ndx, item in enumerate(iter):
        yield current_ndx, item
        if current_ndx == print_ndx:
            duration_sec = ((time.time() - start_ts) / (current_ndx - start_ndx + 1) * (iter_len - start_ndx))
            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)

            log.info(
                f'{desc_str}{current_ndx:-4}/{iter_len},于{str(done_dt).rsplit(".", 1)[0]},{str(done_td).rsplit(".", 1)[0]}完成')

            print_ndx *= backoff

        if current_ndx + 1 == start_ndx:
            start_ts = time.time()
    log.warning(f"{desc_str} ----/{iter_len},于{str(datetime.datetime.now()).rsplit('.', 1)[0]}完成")
