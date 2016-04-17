from attrdict import AttrDict
import h5py
import os

def load_mnist():
    data_dir = os.getenv('MYSHKIN_DATADIR', '.')
    data_file = os.path.join(data_dir, 'mnist.hdf5')

    if not os.path.isfile(data_file):
        raise Exception("file {:s} does not exist ($MYSHKIN_DATADIR={:s})".format(data_file, data_dir))
    else:
        fid = h5py.File(data_file, 'r')

        rval = AttrDict({
            'x_train': fid['x_train'][:],
            'y_train': fid['t_train'][:],
            'x_valid': fid['x_valid'][:],
            'y_valid': fid['t_valid'][:],
            'x_test': fid['x_test'][:],
            'y_test': fid['t_test'][:]
        })

        fid.close()

        return rval
