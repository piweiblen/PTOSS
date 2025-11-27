#include <stdio.h>
#include <cuda.h>

#include "Board.h"
#include "PTOSS_kernel.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

__device__ const int x_around[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
__device__ const int y_around[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
#define AR_IDX(x,y) ((((x)+(3*(y))+5)*4)/5)

// define device functions for working with boards

// these get and set functions are a little tedious, but it'll make life simple if we want to pack the board arrays later
__device__ int get_elem(int* b_arr, int b_idx, int num_b, int elem_idx) {
    return b_arr[elem_idx * num_b + b_idx];
}

__device__ int get_ones_num(int* b_arr, int b_idx, int num_b) {
    return get_elem(b_arr, b_idx, num_b, 0);
}

__device__ int get_ones_left(int* b_arr, int b_idx, int num_b) {
    return get_elem(b_arr, b_idx, num_b, 1);
}

__device__ int get_last_num(int* b_arr, int b_idx, int num_b) {
    return get_elem(b_arr, b_idx, num_b, 2);
}

__device__ int get_pos_x(int* b_arr, int b_idx, int num_b, int arr_idx) {
    return get_elem(b_arr, b_idx, num_b, 3+3*arr_idx);
}

__device__ int get_pos_y(int* b_arr, int b_idx, int num_b, int arr_idx) {
    return get_elem(b_arr, b_idx, num_b, 4+3*arr_idx);
}

__device__ int get_info(int* b_arr, int b_idx, int num_b, int arr_idx) {
    return get_elem(b_arr, b_idx, num_b, 5+3*arr_idx);
}

__device__ void set_elem(int* b_arr, int b_idx, int num_b, int elem_idx, int elem) {
    b_arr[elem_idx * num_b + b_idx] = elem;
}

__device__ void set_ones_num(int* b_arr, int b_idx, int num_b, int elem) {
    set_elem(b_arr, b_idx, num_b, 0, elem);
}

__device__ void set_ones_left(int* b_arr, int b_idx, int num_b, int elem) {
    set_elem(b_arr, b_idx, num_b, 1, elem);
}

__device__ void set_last_num(int* b_arr, int b_idx, int num_b, int elem) {
    set_elem(b_arr, b_idx, num_b, 2, elem);
}

__device__ void set_pos_x(int* b_arr, int b_idx, int num_b, int arr_idx, int elem) {
    set_elem(b_arr, b_idx, num_b, 3+3*arr_idx, elem);
}

__device__ void set_pos_y(int* b_arr, int b_idx, int num_b, int arr_idx, int elem) {
    set_elem(b_arr, b_idx, num_b, 4+3*arr_idx, elem);
}

__device__ void set_info(int* b_arr, int b_idx, int num_b, int arr_idx, int elem) {
    set_elem(b_arr, b_idx, num_b, 5+3*arr_idx, elem);
}

__device__ void copy_device_board(int* b_arr1, int b_idx1, int num_b1, int* b_arr2, int b_idx2, int num_b2) {
    for (int i=0; i<sizeof(Board)/sizeof(int); i++) {
        set_elem(b_arr1, b_idx1, num_b1, i, get_elem(b_arr2, b_idx2, num_b2, i));
    }
}

// these are ports of the functions in Board.cu
__device__ void insert_element(int* b_arr, int b_idx, int num_b, int x, int y, int ones_added) {

    // insert an element onto our board
    int last_num = get_last_num(b_arr, b_idx, num_b);
    set_pos_x(b_arr, b_idx, num_b, last_num, x);
    set_pos_y(b_arr, b_idx, num_b, last_num, y);
    set_info(b_arr, b_idx, num_b, last_num, ones_added);
    set_last_num(b_arr, b_idx, num_b, last_num + 1);
}

__device__ void insert_one(int* b_arr, int b_idx, int num_b, int x, int y) {

    // insert a one onto our board
    int ones_left = get_ones_left(b_arr, b_idx, num_b);
    set_pos_x(b_arr, b_idx, num_b, MAX_HEIGHT-ones_left, x);
    set_pos_y(b_arr, b_idx, num_b, MAX_HEIGHT-ones_left, y);
    set_ones_left(b_arr, b_idx, num_b, ones_left - 1);
}

__device__ void remove_element(int* b_arr, int b_idx, int num_b) {

    // remove an element from our board
    int last_num = get_last_num(b_arr, b_idx, num_b);
    int ones_left = get_ones_left(b_arr, b_idx, num_b);
    set_ones_left(b_arr, b_idx, num_b, get_info(b_arr, b_idx, num_b, last_num - 1));
    set_last_num(b_arr, b_idx, num_b, last_num - 1);
}

__device__ int get_sum(int* b_arr, int b_idx, int num_b, int x, int y, int* least_neighbor) {

    // get the sum of elements surrounding position
    // returns -1 if position already populated
    int sum = 0;
    int last_num = get_last_num(b_arr, b_idx, num_b);
    int ones_left = get_ones_left(b_arr, b_idx, num_b);
    int ones_num = get_ones_num(b_arr, b_idx, num_b);
    int pos_x, pos_y;
    *least_neighbor = MAX_HEIGHT;
    for (int i=0; i<last_num; i++) {
        pos_x = get_pos_x(b_arr, b_idx, num_b, i);
        pos_y = get_pos_y(b_arr, b_idx, num_b, i);
        if ((pos_x == x) && (pos_y == y)) {
            return -1;
        }
        if ((x-1 <= pos_x) && (pos_x <= x+1)) {
            if ((y-1 <= pos_y) && (pos_y <= y+1)) {
                sum += i+2;
                *least_neighbor = MIN(*least_neighbor, i);
            }
        }
    }
    for (int i=MAX_HEIGHT-ones_num; i<MAX_HEIGHT-ones_left; i++) {
        pos_x = get_pos_x(b_arr, b_idx, num_b, i);
        pos_y = get_pos_y(b_arr, b_idx, num_b, i);
        if ((pos_x == x) && (pos_y == y)) {
            return -1;
        }
        if ((x-1 <= pos_x) && (pos_x <= x+1)) {
            if ((y-1 <= pos_y) && (pos_y <= y+1)) {
                sum += 1;
                *least_neighbor = MIN(*least_neighbor, i);
            }
        }
    }
    return sum;
}

__device__ bool look_around(int* b_arr, int b_idx, int num_b, int index, int start_H) {

    // check around the location of a particular index for a spot to place the next element
    int new_x, new_y;
    int min_nb;
    int last_num = get_last_num(b_arr, b_idx, num_b);
    int pos_x = get_pos_x(b_arr, b_idx, num_b, index);
    int pos_y = get_pos_y(b_arr, b_idx, num_b, index);
    for (int H=start_H; H<8; H++) {  // iterate over all spots around (i+2)
        new_x = pos_x + x_around[H];
        new_y = pos_y + y_around[H];
        if (get_sum(b_arr, b_idx, num_b, new_x, new_y, &min_nb) == last_num+2) {
            if (min_nb == index) {  // don't go in spots we've already checked
                insert_element(b_arr, b_idx, num_b, new_x, new_y, 0);
                return true;
            }
        }
    }
    return false;
}

__device__ void next_board_state(int* b_arr, int b_idx, int num_b) {

    // iterate a board to its next state i.e. the next position in the search
    // this function assumes that "2" has already been placed

    int last_num = get_last_num(b_arr, b_idx, num_b);
    int ones_left = get_ones_left(b_arr, b_idx, num_b);
    int ones_num = get_ones_num(b_arr, b_idx, num_b);
    // first try to add the next number
    for (int i=0; i<last_num / 2; i++) {  // choose a num (i+2) to try to place around
        if (look_around(b_arr, b_idx, num_b, i, 0)) return;
    }
    for (int i=last_num - ones_num; i<last_num; i++) {  // we need to also look around high numbers
        if (look_around(b_arr, b_idx, num_b, i, 0)) return;
    }
    if (last_num+2 <= 8) { // we need to also look around ones for small numbers
        for (int i=MAX_HEIGHT-ones_num; i<MAX_HEIGHT-ones_left; i++) { 
            if (look_around(b_arr, b_idx, num_b, i, 0)) return;
        }
    }

    // failing to add a number, we'll attempt to move the current highest to a new position
    // continuing to remove elements until we succeed at moving one
    int old_x, old_y;
    int old_nb, last_H;
    while (true) {
        // first find where we left off
        old_x = get_pos_x(b_arr, b_idx, num_b, last_num - 1);
        old_y = get_pos_y(b_arr, b_idx, num_b, last_num - 1);
        get_sum(b_arr, b_idx, num_b, old_x, old_y, &old_nb);  // TODO being lazy here, should write a new func that stops 
        last_H = AR_IDX(old_x - get_pos_x(b_arr, b_idx, num_b, old_nb), old_y - get_pos_y(b_arr, b_idx, num_b, old_nb));
        // remove the element and search for new spot
        remove_element(b_arr, b_idx, num_b);
        last_num = get_last_num(b_arr, b_idx, num_b);
        ones_left = get_ones_left(b_arr, b_idx, num_b);
        // start with element it was already around
        if (look_around(b_arr, b_idx, num_b, old_nb, last_H+1)) return;
        for (int i=old_nb+1; i<last_num / 2; i++) {  // choose a num (i+2) to try to place around
            if (look_around(b_arr, b_idx, num_b, i, 0)) return;
        }
        for (int i=MAX(old_nb+1, last_num - ones_num); i<last_num; i++) {  // we need to also look around high numbers
            if (look_around(b_arr, b_idx, num_b, i, 0)) return;
        }
        if (last_num+2 <= 8) { // we need to also look around ones for small numbers
            for (int i=MAX(old_nb+1, MAX_HEIGHT-ones_num); i<MAX_HEIGHT-ones_left; i++) { 
                if (look_around(b_arr, b_idx, num_b, i, 0)) return;
            }
        }
    }

    // big TODO here still: need to be able to insert ones along with elems lol

}

// Matrix multiplication kernel thread specificationa
__global__ void SearchKernel(int* b_arr, int* max_board, int* cur_max, int num_b) {
    //TODO assign each thread one board to completely search the state space of
    int b_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (b_idx < num_b) {
        int anchor_x = get_pos_x(b_arr, b_idx, num_b, get_last_num(b_arr, b_idx, num_b) - 1);
        int anchor_y = get_pos_y(b_arr, b_idx, num_b, get_last_num(b_arr, b_idx, num_b) - 1);
        atomicMax(cur_max, get_last_num(b_arr, b_idx, num_b));
        while (true) {
            next_board_state(b_arr, b_idx, num_b);
            // if (*cur_max < get_last_num(b_arr, b_idx, num_b)) {
            //     atomicMax(cur_max, get_last_num(b_arr, b_idx, num_b));
            //     copy_device_board(max_board, 0, 1, b_arr, b_idx, num_b);  // TODO make atomic??
            // }
            if ((anchor_x != get_pos_x(b_arr, b_idx, num_b, get_last_num(b_arr, b_idx, num_b) - 1))
             || (anchor_y != get_pos_y(b_arr, b_idx, num_b, get_last_num(b_arr, b_idx, num_b) - 1))) {
                break;
            }
        }
    }
}

// Kernel calling function specification
void SearchOnDevice(Board* Boards, Board* max_board, int num_b) {

    // do a bit of data marshalling
    int* Boards_host = (int*) malloc(num_b * sizeof(Board));
    flatten_board_list(Boards_host, Boards, num_b);
    printf("host last num: %d\n", Boards_host[1 * num_b + 3]);
    

    // set up device memory
    int* cur_max;
    cudaMalloc((void**) &cur_max, sizeof(int));
	cudaMemset(cur_max, 0, sizeof(int));
    int* max_board_device;
    cudaMalloc((void**) &max_board_device, sizeof(Board));
    int* Boards_device;
    cudaMalloc((void**) &Boards_device, num_b * sizeof(Board));
    cudaMemcpy(Boards_device, Boards_host, num_b * sizeof(Board), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

    // Setup the execution configuration
	dim3 blockDim(BLOCK_SIZE, 1, 1);
	dim3 gridDim((num_b + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);

    // Launch the device computation threads
	SearchKernel<<<gridDim,blockDim>>>(Boards_device, max_board_device, cur_max, num_b);
	cudaDeviceSynchronize();

    int host_max;
    cudaMemcpy(&host_max, cur_max, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
    printf("max found: %d\n", host_max);

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
