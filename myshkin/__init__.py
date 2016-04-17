import os

if 'guppy' in os.uname()[1]:
    from util.gpu import get_board
    board = get_board()
    print "using board: {:d}".format(board)

    os.environ['TENSORFLOW_DEVICE'] = "/gpu:{:d}".format(board)
