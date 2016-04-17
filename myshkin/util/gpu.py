# get board
import os
import gpu_lock2 as gg

def get_board():
    board_id = gg.obtain_lock_id()
    if board_id < 0:
        raise Exception("could not obtain lock")
    else:
        return board_id
