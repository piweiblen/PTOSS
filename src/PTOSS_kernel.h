#ifndef _GPU_KERNEL_
#define _GPU_KERNEL_

#define BLOCK_SIZE 32

void LaunchSearchKernel(Board* Boards, uint32_t** Boards_device, int** cur_max, int num_b);
Board GetKernelResult(uint32_t** Boards_device, int** cur_max, int num_b);

#endif // _GPU_KERNEL_

