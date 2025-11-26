#include <stdio.h>
#include <cuda.h>

#include "Board.h"
#include "PTOSS_kernel.h"


// define device functions for working with boards
__device__ int get_elem(int* flat_boards, int elem_idx, int b_idx, int num_boards) {
    return flat_boards[elem_idx * num_boards + b_idx];
}

__device__ int get_ones_num(int* flat_boards, int b_idx, int num_boards) {
    return get_elem(flat_boards, 0, b_idx, num_boards);
}

__device__ int get_ones_left(int* flat_boards, int b_idx, int num_boards) {
    return get_elem(flat_boards, 1, b_idx, num_boards);
}

__device__ int get_last_num(int* flat_boards, int b_idx, int num_boards) {
    return get_elem(flat_boards, 2, b_idx, num_boards);
}

__device__ int get_pos_x(int* flat_boards, int arr_idx, int b_idx, int num_boards) {
    return get_elem(flat_boards, 3+3*arr_idx, b_idx, num_boards);
}

__device__ int get_pos_y(int* flat_boards, int arr_idx, int b_idx, int num_boards) {
    return get_elem(flat_boards, 4+3*arr_idx, b_idx, num_boards);
}

__device__ int get_info(int* flat_boards, int arr_idx, int b_idx, int num_boards) {
    return get_elem(flat_boards, 5+3*arr_idx, b_idx, num_boards);
}

// TODO need to port all of this to device code
// obviously this is stuff is still work in progress too, but I wanna get code on the GPU before things get overwhelmingly complex

// void insert_element(Board* B, int x, int y, int ones_added) {

//     // insert an element onto our board
//     B->pos_x[B->last_num] = x;
//     B->pos_y[B->last_num] = y;
//     B->info[B->last_num] = ones_added;
//     B->last_num += 1;
// }

// void insert_one(Board* B, int x, int y) {

//     // insert a one onto our board
//     B->pos_x[MAX_HEIGHT-B->ones_left] = x;
//     B->pos_y[MAX_HEIGHT-B->ones_left] = y;
//     B->ones_left -= 1;
// }

// void remove_element(Board* B) {

//     // remove an element from our board
//     B->ones_left += B->info[B->last_num - 1];
//     B->last_num -= 1;
// }

// int get_sum(Board* B, int x, int y, int* least_neighbor) {  //TODO lots of wasted time here, in the end we'll want to stop when we know (sum > target)

//     // get the sum of elements surrounding position
//     // returns -1 if position already populated
//     int sum = 0;
//     *least_neighbor = MAX_HEIGHT;
//     for (int i=0; i<B->last_num; i++) {
//         if ((B->pos_x[i] == x) && (B->pos_y[i] == y)) {
//             return -1;
//         }
//         if ((x-1 <= B->pos_x[i]) && (B->pos_x[i] <= x+1)) {
//             if ((y-1 <= B->pos_y[i]) && (B->pos_y[i] <= y+1)) {
//                 sum += i+2;
//                 *least_neighbor = MIN(*least_neighbor, i);
//             }
//         }
//     }
//     for (int i=MAX_HEIGHT-B->ones_num; i<MAX_HEIGHT-B->ones_left; i++) {
//         if ((B->pos_x[i] == x) && (B->pos_y[i] == y)) {
//             return -1;
//         }
//         if ((x-1 <= B->pos_x[i]) && (B->pos_x[i] <= x+1)) {
//             if ((y-1 <= B->pos_y[i]) && (B->pos_y[i] <= y+1)) {
//                 sum += 1;
//                 *least_neighbor = MIN(*least_neighbor, i);
//             }
//         }
//     }
//     return sum;
// }

// bool look_around(Board* B, int index, int start_H) {

//     // check around the location of a particular index for a spot to place the next element
//     int new_x, new_y;
//     int min_nb;
//     for (int H=start_H; H<8; H++) {  // iterate over all spots around (i+2)
//         new_x = B->pos_x[index] + x_around[H];
//         new_y = B->pos_y[index] + y_around[H];
//         if (get_sum(B, new_x, new_y, &min_nb) == B->last_num+2) {
//             if (min_nb == index) {  // don't go in spots we've already checked
//                 insert_element(B, new_x, new_y, 0);
//                 return true;
//             }
//         }
//     }
//     return false;
// }

// void next_board_state(Board* B) {

//     // iterate a board to its next state i.e. the next position in the search
//     // this function assumes that "2" has already been placed

//     // first try to add the next number
//     for (int i=0; i<B->last_num / 2; i++) {  // choose a num (i+2) to try to place around
//         if (look_around(B, i, 0)) return;
//     }
//     for (int i=B->last_num - B->ones_num; i<B->last_num; i++) {  // we need to also look around high numbers
//         if (look_around(B, i, 0)) return;
//     }
//     if (B->last_num+2 <= 8) { // we need to also look around ones for small numbers
//         for (int i=MAX_HEIGHT-B->ones_num; i<MAX_HEIGHT-B->ones_left; i++) { 
//             if (look_around(B, i, 0)) return;
//         }
//     }

//     // failing to add a number, we'll attempt to move the current highest to a new position
//     // continuing to remove elements until we succeed at moving one
//     int old_x, old_y;
//     int old_nb, last_H;
//     while (true) {
//         // first find where we left off
//         old_x = B->pos_x[B->last_num - 1];
//         old_y = B->pos_y[B->last_num - 1];
//         get_sum(B, old_x, old_y, &old_nb);  // TODO being lazy here, should write a new func that stops 
//         last_H = AR_IDX(old_x - B->pos_x[old_nb], old_y - B->pos_y[old_nb]);
//         // remove the element and search for new spot
//         remove_element(B);
//         // TODO dry out this WET code
//         // start with element it was already around
//         look_around(B, old_nb, last_H+1);
//         for (int i=old_nb+1; i<B->last_num / 2; i++) {  // choose a num (i+2) to try to place around
//             if (look_around(B, i, 0)) return;
//         }
//         for (int i=MAX(old_nb+1, B->last_num - B->ones_num); i<B->last_num; i++) {  // we need to also look around high numbers
//             if (look_around(B, i, 0)) return;
//         }
//         if (B->last_num+2 <= 8) { // we need to also look around ones for small numbers
//             for (int i=MAX(old_nb+1, MAX_HEIGHT-B->ones_num); i<MAX_HEIGHT-B->ones_left; i++) { 
//                 if (look_around(B, i, 0)) return;
//             }
//         }
//     }

//     // big TODO here still: need to be able to insert ones along with elems lol

// }

// Matrix multiplication kernel thread specificationa
__global__ void SearchKernel(int* flat_boards, int* max_board, int* cur_max, int num_boards) {
    //TODO assign each thread one board to completely search the state space of
    int b_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (b_idx < num_boards) {
        int anchor_x = get_pos_x(flat_boards, get_last_num(flat_boards, b_idx, num_boards) - 1, b_idx, num_boards);
        int anchor_y = get_pos_y(flat_boards, get_last_num(flat_boards, b_idx, num_boards) - 1, b_idx, num_boards);
        while (true) {
            // next_board_state(flat_boards);  // TODO use device nbs func
            if (*cur_max < get_last_num(flat_boards, b_idx, num_boards)) {
                atomicMax(cur_max, get_last_num(flat_boards, b_idx, num_boards));;
                // CopyHostBoard(&max_board, &search_board);  // TODO new atomic device copy func needed
            }
            if ((anchor_x != get_pos_x(flat_boards, get_last_num(flat_boards, b_idx, num_boards) - 1, b_idx, num_boards))
             || (anchor_y != get_pos_y(flat_boards, get_last_num(flat_boards, b_idx, num_boards) - 1, b_idx, num_boards))) {
                break;
            }
        }
    }
}

// Kernel calling function specification
void SearchOnDevice(Board* Boards, Board* max_board, int num_boards) {

    // do a bit of data marshalling
    int* Boards_host = (int*) malloc(num_boards * sizeof(Board));
    flatten_board_list(Boards_host, Boards, num_boards);

    // set up device memory
    int* cur_max;
    cudaMalloc((void**) &cur_max, sizeof(int));
    int* max_board_device;
    cudaMalloc((void**) &max_board_device, sizeof(Board));
    int* Boards_device;
    cudaMalloc((void**) &Boards_device, num_boards * sizeof(Board));
    cudaMemcpy(Boards_device, Boards_host, num_boards * sizeof(Board), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

    // Setup the execution configuration
	dim3 blockDim(BLOCK_SIZE, 1, 1);
	dim3 gridDim((num_boards + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);

    // Launch the device computation threads
	SearchKernel<<<gridDim,blockDim>>>(Boards_device, max_board_device, cur_max, num_boards);
	cudaDeviceSynchronize();

    // copy results back to host
    int* max_board_host = (int*) malloc(sizeof(Board));
    cudaMemcpy(max_board_host, max_board_device, sizeof(Board), cudaMemcpyDeviceToHost);
    unflatten_board_list(max_board, max_board_host, 1);

    // free memory
    free(Boards_host);
    cudaFree(cur_max);
    cudaFree(max_board_device);
    cudaFree(Boards_device);
}
