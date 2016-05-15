import numpy as np

def patch_view(arr, n_cols, border_width=1):
    n_images, h, w, d = arr.shape
    n_rows = int(np.ceil(1.0 * n_images / n_cols))

    rval = np.zeros((n_rows * (h + border_width) + border_width + 1,
                     n_cols * (w + border_width) + border_width, d))
    rval[-1:, :, :] = 1.0

    for ind in xrange(n_images):
        i = int(np.floor(ind / n_cols)) * (h + border_width) + border_width
        j = (ind % n_cols) * (w + border_width) + border_width
        rval[i:i+h, j:j+w, :] = arr[ind]

    if rval.shape[2] == 1:
        rval = rval[:, :, 0]

    return rval

def interleave(arrs):
    for i in xrange(len(arrs)):
        if len(arrs[i].shape) == 3:
            arrs[i] = np.tile(arrs[i][..., None], (1, 1, 1, 3))

    arr_shape = arrs[0].shape
    for i in xrange(1, len(arrs)):
        assert arrs[i].shape == arr_shape

    rval = np.empty((arr_shape[0] * len(arrs),) + arr_shape[1:])

    for j in xrange(arr_shape[0]):
        for (i, arr) in enumerate(arrs):
            rval[j * len(arrs) + i] = arr[j]

    return rval
