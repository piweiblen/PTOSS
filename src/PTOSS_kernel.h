#ifndef _GPU_KERNEL_
#define _GPU_KERNEL_

#define BLOCK_SIZE 32

void SearchOnDevice(Board* Boards, Board* max_board, int num_boards);

#endif // _GPU_KERNEL_

