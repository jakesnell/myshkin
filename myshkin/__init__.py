import os
import gpu_lock2 as gg

if len(gg.board_ids()) > 0:
    from util.gpu import get_board
    board = get_board()
    print "using board: {:d}".format(board)

    os.environ['TENSORFLOW_DEVICE'] = "/gpu:{:d}".format(board)
