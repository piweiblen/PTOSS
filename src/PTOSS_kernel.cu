#include <stdio.h>
#include <cuda.h>

#include "Board.h"
#include "PTOSS_kernel.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

__device__ const int x_around[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
__device__ const int y_around[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
#define AR_IDX(x,y) ((((x)+(3*(y))+5)*4)/5)

__device__ const int P_rngs[10] = {0, 1, 9, 37, 93, 163, 219, 247, 255, 256};
__device__ const int P_bits[256] = {0, 1, 2, 4, 8, 16, 32, 64, 128, 3, 5, 9, 17, 33, 65, 129, 6, 10, 18, 34, 66, 130, 12, 20, 36, 68, 132, 24, 40, 72, 136, 48, 80, 144, 96, 160, 192, 7, 11, 19, 35, 67, 131, 13, 21, 37, 69, 133, 25, 41, 73, 137, 49, 81, 145, 97, 161, 193, 14, 22, 38, 70, 134, 26, 42, 74, 138, 50, 82, 146, 98, 162, 194, 28, 44, 76, 140, 52, 84, 148, 100, 164, 196, 56, 88, 152, 104, 168, 200, 112, 176, 208, 224, 15, 23, 39, 71, 135, 27, 43, 75, 139, 51, 83, 147, 99, 163, 195, 29, 45, 77, 141, 53, 85, 149, 101, 165, 197, 57, 89, 153, 105, 169, 201, 113, 177, 209, 225, 30, 46, 78, 142, 54, 86, 150, 102, 166, 198, 58, 90, 154, 106, 170, 202, 114, 178, 210, 226, 60, 92, 156, 108, 172, 204, 116, 180, 212, 228, 120, 184, 216, 232, 240, 31, 47, 79, 143, 55, 87, 151, 103, 167, 199, 59, 91, 155, 107, 171, 203, 115, 179, 211, 227, 61, 93, 157, 109, 173, 205, 117, 181, 213, 229, 121, 185, 217, 233, 241, 62, 94, 158, 110, 174, 206, 118, 182, 214, 230, 122, 186, 218, 234, 242, 124, 188, 220, 236, 244, 248, 63, 95, 159, 111, 175, 207, 119, 183, 215, 231, 123, 187, 219, 235, 243, 125, 189, 221, 237, 245, 249, 126, 190, 222, 238, 246, 250, 252, 127, 191, 223, 239, 247, 251, 253, 254, 255};
__device__ const int P_idxs[256] = {0, 1, 2, 9, 3, 10, 16, 37, 4, 11, 17, 38, 22, 43, 58, 93, 5, 12, 18, 39, 23, 44, 59, 94, 27, 48, 63, 98, 73, 108, 128, 163, 6, 13, 19, 40, 24, 45, 60, 95, 28, 49, 64, 99, 74, 109, 129, 164, 31, 52, 67, 102, 77, 112, 132, 167, 83, 118, 138, 173, 148, 183, 198, 219, 7, 14, 20, 41, 25, 46, 61, 96, 29, 50, 65, 100, 75, 110, 130, 165, 32, 53, 68, 103, 78, 113, 133, 168, 84, 119, 139, 174, 149, 184, 199, 220, 34, 55, 70, 105, 80, 115, 135, 170, 86, 121, 141, 176, 151, 186, 201, 222, 89, 124, 144, 179, 154, 189, 204, 225, 158, 193, 208, 229, 213, 234, 240, 247, 8, 15, 21, 42, 26, 47, 62, 97, 30, 51, 66, 101, 76, 111, 131, 166, 33, 54, 69, 104, 79, 114, 134, 169, 85, 120, 140, 175, 150, 185, 200, 221, 35, 56, 71, 106, 81, 116, 136, 171, 87, 122, 142, 177, 152, 187, 202, 223, 90, 125, 145, 180, 155, 190, 205, 226, 159, 194, 209, 230, 214, 235, 241, 248, 36, 57, 72, 107, 82, 117, 137, 172, 88, 123, 143, 178, 153, 188, 203, 224, 91, 126, 146, 181, 156, 191, 206, 227, 160, 195, 210, 231, 215, 236, 242, 249, 92, 127, 147, 182, 157, 192, 207, 228, 161, 196, 211, 232, 216, 237, 243, 250, 162, 197, 212, 233, 217, 238, 244, 251, 218, 239, 245, 252, 246, 253, 254, 255};
__device__ const int P_wieght[256] = {0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8};

// define device functions for working with boards

// these get and set functions are a little tedious, but it'll make life simple if we want to pack the board arrays later
__device__ uint32_t get_elem(uint32_t* b_arr, int b_idx, int num_b, int elem_idx) {
    return b_arr[elem_idx * num_b + b_idx];
}

__device__ int get_ones_num(uint32_t* b_arr, int b_idx, int num_b) {
    return get_elem(b_arr, b_idx, num_b, 0);
}

__device__ int get_ones_left(uint32_t* b_arr, int b_idx, int num_b) {
    return get_elem(b_arr, b_idx, num_b, 1);
}

__device__ int get_last_num(uint32_t* b_arr, int b_idx, int num_b) {
    return get_elem(b_arr, b_idx, num_b, 2);
}

__device__ int get_pos_x(uint32_t* b_arr, int b_idx, int num_b, int arr_idx) {
    uint32_t packed_int = get_elem(b_arr, b_idx, num_b, 3+arr_idx);
    return unpack_pos_x(packed_int);
}

__device__ int get_pos_y(uint32_t* b_arr, int b_idx, int num_b, int arr_idx) {
    uint32_t packed_int = get_elem(b_arr, b_idx, num_b, 3+arr_idx);
    return unpack_pos_y(packed_int);
}

__device__ int get_info(uint32_t* b_arr, int b_idx, int num_b, int arr_idx) {
    uint32_t packed_int = get_elem(b_arr, b_idx, num_b, 3+arr_idx);
    return unpack_info(packed_int);
}

__device__ void get_pos(uint32_t* b_arr, int b_idx, int num_b, int arr_idx, int* x, int* y) {
    uint32_t packed_int = get_elem(b_arr, b_idx, num_b, 3+arr_idx);
    *x = unpack_pos_x(packed_int);
    *y = unpack_pos_y(packed_int);
}

__device__ void get_pos_info(uint32_t* b_arr, int b_idx, int num_b, int arr_idx, int* x, int* y, int* info) {
    uint32_t packed_int = get_elem(b_arr, b_idx, num_b, 3+arr_idx);
    *x = unpack_pos_x(packed_int);
    *y = unpack_pos_y(packed_int);
    *info = unpack_info(packed_int);
}

__device__ void set_elem(uint32_t* b_arr, int b_idx, int num_b, int elem_idx, uint32_t elem) {
    b_arr[elem_idx * num_b + b_idx] = elem;
}

__device__ void set_ones_num(uint32_t* b_arr, int b_idx, int num_b, int new_ones_num) {
    set_elem(b_arr, b_idx, num_b, 0, new_ones_num);
}

__device__ void set_ones_left(uint32_t* b_arr, int b_idx, int num_b, int new_ones_left) {
    set_elem(b_arr, b_idx, num_b, 1, new_ones_left);
}

__device__ void set_last_num(uint32_t* b_arr, int b_idx, int num_b, int elem) {
    set_elem(b_arr, b_idx, num_b, 2, elem);
}

__device__ void set_pos_x(uint32_t* b_arr, int b_idx, int num_b, int arr_idx, int new_x) {
    uint32_t packed_int = pack_pos_x(get_elem(b_arr, b_idx, num_b, 3+arr_idx), new_x);
    set_elem(b_arr, b_idx, num_b, 3+arr_idx, packed_int);
}

__device__ void set_pos_y(uint32_t* b_arr, int b_idx, int num_b, int arr_idx, int new_y) {
    uint32_t packed_int = pack_pos_y(get_elem(b_arr, b_idx, num_b, 3+arr_idx), new_y);
    set_elem(b_arr, b_idx, num_b, 3+arr_idx, packed_int);
}

__device__ void set_info(uint32_t* b_arr, int b_idx, int num_b, int arr_idx, int new_info) {
    uint32_t packed_int = pack_info(get_elem(b_arr, b_idx, num_b, 3+arr_idx), new_info);
    set_elem(b_arr, b_idx, num_b, 3+arr_idx, packed_int);
}

__device__ void set_pos(uint32_t* b_arr, int b_idx, int num_b, int arr_idx, int new_x, int new_y) {
    uint32_t packed_int = pack_pos(get_elem(b_arr, b_idx, num_b, 3+arr_idx), new_x, new_y);
    set_elem(b_arr, b_idx, num_b, 3+arr_idx, packed_int);
}

__device__ void set_pos_info(uint32_t* b_arr, int b_idx, int num_b, int arr_idx, int new_x, int new_y, int new_info) {
    uint32_t packed_int = pack(new_x, new_y, new_info);
    set_elem(b_arr, b_idx, num_b, 3+arr_idx, packed_int);
}

__device__ void copy_device_board(uint32_t* b_arr1, int b_idx1, int num_b1, uint32_t* b_arr2, int b_idx2, int num_b2) {
    for (int i=0; i<sizeof(Board)/sizeof(int); i++) {
        set_elem(b_arr1, b_idx1, num_b1, i, get_elem(b_arr2, b_idx2, num_b2, i));
    }
}

// these are ports of the functions in Board.cu
__device__ void insert_element(uint32_t* b_arr, int b_idx, int num_b, int x, int y, int ones_added) {

    // insert an element onto our board
    int last_num = get_last_num(b_arr, b_idx, num_b);
    set_pos_info(b_arr, b_idx, num_b, last_num, x, y, ones_added);
    set_last_num(b_arr, b_idx, num_b, last_num + 1);
}

__device__ void insert_one(uint32_t* b_arr, int b_idx, int num_b, int x, int y) {

    // insert a one onto our board
    int ones_left = get_ones_left(b_arr, b_idx, num_b);
    set_pos(b_arr, b_idx, num_b, MAX_HEIGHT-ones_left, x, y);
    set_ones_left(b_arr, b_idx, num_b, ones_left - 1);
}

__device__ void remove_element(uint32_t* b_arr, int b_idx, int num_b, int* one_positions) {

    // remove an element from our board
    int last_num = get_last_num(b_arr, b_idx, num_b);
    int ones_left = get_ones_left(b_arr, b_idx, num_b);
    int x, y, otr;
    get_pos_info(b_arr, b_idx, num_b, last_num - 1, &x, &y, &otr);
    *one_positions = 0;
    for (int i=MAX_HEIGHT-ones_left-otr; i<MAX_HEIGHT-ones_left; i++) {
        int xi, yi;
        get_pos(b_arr, b_idx, num_b, i, &xi, &yi);
        *one_positions |= (1 << AR_IDX(xi - x, yi - y));
    }
    set_ones_left(b_arr, b_idx, num_b, ones_left + otr);
    set_last_num(b_arr, b_idx, num_b, last_num - 1);
}

__device__ int get_sum(uint32_t* b_arr, int b_idx, int num_b, int x, int y, int* least_neighbor, int* open_neighbors) {

    // get the sum of elements surrounding position
    // returns -1 if position already populated
    int sum = 0;
    int last_num = get_last_num(b_arr, b_idx, num_b);
    int ones_left = get_ones_left(b_arr, b_idx, num_b);
    int ones_num = get_ones_num(b_arr, b_idx, num_b);
    int pos_x, pos_y;
    *least_neighbor = MAX_HEIGHT;
    *open_neighbors = 255;
    for (int i=0; i<last_num; i++) {
        get_pos(b_arr, b_idx, num_b, i, &pos_x, &pos_y);
        if ((pos_x == x) && (pos_y == y)) {
            return -1;
        }
        if ((x-1 <= pos_x) && (pos_x <= x+1)) {
            if ((y-1 <= pos_y) && (pos_y <= y+1)) {
                sum += i+2;
                *least_neighbor = MIN(*least_neighbor, i);
                *open_neighbors ^= (1 << AR_IDX(pos_x - x, pos_y - y));
            }
        }
    }
    for (int i=MAX_HEIGHT-ones_num; i<MAX_HEIGHT-ones_left; i++) {
        get_pos(b_arr, b_idx, num_b, i, &pos_x, &pos_y);
        if ((pos_x == x) && (pos_y == y)) {
            return -1;
        }
        if ((x-1 <= pos_x) && (pos_x <= x+1)) {
            if ((y-1 <= pos_y) && (pos_y <= y+1)) {
                sum += 1;
                *least_neighbor = MIN(*least_neighbor, i);
                *open_neighbors ^= (1 << AR_IDX(pos_x - x, pos_y - y));
            }
        }
    }
    return sum;
}

__device__ bool look_around(uint32_t* b_arr, int b_idx, int num_b, int index, int start_H, int start_P) {

    // check around the location of a particular index for a spot to place the next element
    int new_x, new_y;
    int cur_sum, min_nb, open_nb, ones_needed;
    int last_num = get_last_num(b_arr, b_idx, num_b);
    int pos_x, pos_y;
    get_pos(b_arr, b_idx, num_b, index, &pos_x, &pos_y);
    for (int H=start_H; H<8; H++) {  // iterate over all spots around (i+2)
        new_x = pos_x + x_around[H];
        new_y = pos_y + y_around[H];
        cur_sum = get_sum(b_arr, b_idx, num_b, new_x, new_y, &min_nb, &open_nb);
        if (min_nb == index) {  // don't go in spots we've already checked
            if (cur_sum <= last_num + 2) {
                ones_needed = last_num + 2 - cur_sum;
                if (ones_needed <= MIN(get_ones_left(b_arr, b_idx, num_b), P_wieght[P_idxs[open_nb]])) {
                    for (int P=MAX(start_P, P_rngs[ones_needed]); P<P_rngs[ones_needed+1]; P++) {
                        if ((P_bits[P] & open_nb) != P_bits[P]) continue;  // one positions must be open
                        insert_element(b_arr, b_idx, num_b, new_x, new_y, ones_needed);
                        for (int b=0; b<8; b++) {
                            if (P_bits[P] & (1<<b)) insert_one(b_arr, b_idx, num_b, new_x+x_around[b], new_y+y_around[b]);
                        }
                        return true;
                    }
                }
            }
        }
        start_P = 0;
    }
    return false;
}

__device__ void next_board_state(uint32_t* b_arr, int b_idx, int num_b) {

    // iterate a board to its next state i.e. the next position in the search
    // this function assumes that "2" has already been placed

    int last_num = get_last_num(b_arr, b_idx, num_b);
    int ones_left = get_ones_left(b_arr, b_idx, num_b);
    int ones_num = get_ones_num(b_arr, b_idx, num_b);
    // first try to add the next number
    for (int i=0; i<last_num / 2; i++) {  // choose a num (i+2) to try to place around
        if (look_around(b_arr, b_idx, num_b, i, 0, 0)) return;
    }
    for (int i=last_num - ones_num; i<last_num; i++) {  // we need to also look around high numbers
        if (look_around(b_arr, b_idx, num_b, i, 0, 0)) return;
    }
    if (last_num+2 <= 8) { // we need to also look around ones for small numbers
        for (int i=MAX_HEIGHT-ones_num; i<MAX_HEIGHT-ones_left; i++) { 
            if (look_around(b_arr, b_idx, num_b, i, 0, 0)) return;
        }
    }

    // failing to add a number, we'll attempt to move the current highest to a new position
    // continuing to remove elements until we succeed at moving one
    int old_x, old_y;
    int old_nb, last_H, last_P;
    while (last_num - 1) {  // abort if "3" is removed
        // first find where we left off
        get_pos(b_arr, b_idx, num_b, last_num - 1, &old_x, &old_y);
        // old_x = get_pos_x(b_arr, b_idx, num_b, last_num - 1);
        // old_y = get_pos_y(b_arr, b_idx, num_b, last_num - 1);
        get_sum(b_arr, b_idx, num_b, old_x, old_y, &old_nb, &last_P);  // this line is only to get old_nb, last_P is garbage data here
                                                                       // TODO: stop being lazy here, should write a seperate func that stops when old_nb is found
        last_H = AR_IDX(old_x - get_pos_x(b_arr, b_idx, num_b, old_nb), old_y - get_pos_y(b_arr, b_idx, num_b, old_nb));
        // remove the element and search for new spot
        remove_element(b_arr, b_idx, num_b, &last_P);
        last_num = get_last_num(b_arr, b_idx, num_b);
        ones_left = get_ones_left(b_arr, b_idx, num_b);
        // start with element it was already around
        if (look_around(b_arr, b_idx, num_b, old_nb, last_H, P_idxs[last_P] + 1)) return;
        for (int i=old_nb+1; i<last_num / 2; i++) {  // choose a num (i+2) to try to place around
            if (look_around(b_arr, b_idx, num_b, i, 0, 0)) return;
        }
        for (int i=MAX(old_nb+1, last_num - ones_num); i<last_num; i++) {  // we need to also look around high numbers
            if (look_around(b_arr, b_idx, num_b, i, 0, 0)) return;
        }
        if (last_num+2 <= 8) { // we need to also look around ones for small numbers
            for (int i=MAX(old_nb+1, MAX_HEIGHT-ones_num); i<MAX_HEIGHT-ones_left; i++) { 
                if (look_around(b_arr, b_idx, num_b, i, 0, 0)) return;
            }
        }
    }
}

// Main kernel for parallel board search
__global__ void SearchKernel(uint32_t* b_arr, uint32_t* max_board, int* cur_max, int num_b) {
    //TODO: assign each thread one board to completely search the state space of
    int b_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (b_idx < num_b) {
        int anchor_idx = get_last_num(b_arr, b_idx, num_b) - 1;
        int anchor_x, anchor_y;
        get_pos(b_arr, b_idx, num_b, anchor_idx, &anchor_x, &anchor_y);
        while (true) {
            next_board_state(b_arr, b_idx, num_b);
            if (*cur_max < get_last_num(b_arr, b_idx, num_b)) {
                atomicMax(cur_max, get_last_num(b_arr, b_idx, num_b));
                copy_device_board(max_board, 0, 1, b_arr, b_idx, num_b);  // TODO: make atomic!!
            }
            int test_x, test_y;
            get_pos(b_arr, b_idx, num_b, anchor_idx, &test_x, &test_y);
            if ((anchor_x != test_x)
             || (anchor_y != test_y)) {
                break;
            }
        }
    }
}

// Kernel calling function specification
void SearchOnDevice(Board* Boards, Board* max_board, int num_b) {

    // do a bit of data marshalling
    uint32_t* Boards_host = (uint32_t*) malloc(num_b * (MAX_HEIGHT+3) * sizeof(uint32_t));
    flatten_board_list(Boards_host, Boards, num_b);

    // set up device memory
    int* cur_max;
    cudaMalloc((void**) &cur_max, sizeof(int));
	cudaMemset(cur_max, 0, sizeof(int));
    uint32_t* max_board_device;
    cudaMalloc((void**) &max_board_device, (MAX_HEIGHT+3) * sizeof(uint32_t));
    uint32_t* Boards_device;
    cudaMalloc((void**) &Boards_device, num_b * (MAX_HEIGHT + 3) * sizeof(uint32_t));
    cudaMemcpy(Boards_device, Boards_host, num_b * (MAX_HEIGHT+3) * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Setup the execution configuration
	dim3 blockDim(BLOCK_SIZE, 1, 1);
	dim3 gridDim((num_b + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);

    // Launch the device computation threads
	SearchKernel<<<gridDim,blockDim>>>(Boards_device, max_board_device, cur_max, num_b);
	cudaDeviceSynchronize();

    // copy results back to host
    // int* host_cur_max = (int*) malloc(sizeof(int));
    // cudaMemcpy(host_cur_max, cur_max, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("device max last pos: %d\n", *host_cur_max);
    uint32_t* max_board_host = (uint32_t*) malloc((MAX_HEIGHT+3) * sizeof(uint32_t));
    cudaMemcpy(max_board_host, max_board_device, (MAX_HEIGHT+3) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    unflatten_board_list(max_board, max_board_host, 1);

    // free memory
    free(Boards_host);
    cudaFree(cur_max);
    cudaFree(max_board_device);
    cudaFree(Boards_device);
}
