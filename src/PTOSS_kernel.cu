#include <stdio.h>
#include <cuda.h>

#include "Board.h"
#include "PTOSS_kernel.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

__constant__ int x_around[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
__constant__ int y_around[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
#define AR_IDX(x,y) ((((x)+(3*(y))+5)*4)/5)

__constant__ int P_rngs[10] = {0, 1, 9, 37, 93, 163, 219, 247, 255, 256};
__constant__ int P_bits[256] = {0, 1, 2, 4, 8, 16, 32, 64, 128, 3, 5, 9, 17, 33, 65, 129, 6, 10, 18, 34, 66, 130, 12, 20, 36, 68, 132, 24, 40, 72, 136, 48, 80, 144, 96, 160, 192, 7, 11, 19, 35, 67, 131, 13, 21, 37, 69, 133, 25, 41, 73, 137, 49, 81, 145, 97, 161, 193, 14, 22, 38, 70, 134, 26, 42, 74, 138, 50, 82, 146, 98, 162, 194, 28, 44, 76, 140, 52, 84, 148, 100, 164, 196, 56, 88, 152, 104, 168, 200, 112, 176, 208, 224, 15, 23, 39, 71, 135, 27, 43, 75, 139, 51, 83, 147, 99, 163, 195, 29, 45, 77, 141, 53, 85, 149, 101, 165, 197, 57, 89, 153, 105, 169, 201, 113, 177, 209, 225, 30, 46, 78, 142, 54, 86, 150, 102, 166, 198, 58, 90, 154, 106, 170, 202, 114, 178, 210, 226, 60, 92, 156, 108, 172, 204, 116, 180, 212, 228, 120, 184, 216, 232, 240, 31, 47, 79, 143, 55, 87, 151, 103, 167, 199, 59, 91, 155, 107, 171, 203, 115, 179, 211, 227, 61, 93, 157, 109, 173, 205, 117, 181, 213, 229, 121, 185, 217, 233, 241, 62, 94, 158, 110, 174, 206, 118, 182, 214, 230, 122, 186, 218, 234, 242, 124, 188, 220, 236, 244, 248, 63, 95, 159, 111, 175, 207, 119, 183, 215, 231, 123, 187, 219, 235, 243, 125, 189, 221, 237, 245, 249, 126, 190, 222, 238, 246, 250, 252, 127, 191, 223, 239, 247, 251, 253, 254, 255};
__constant__ int P_idxs[256] = {0, 1, 2, 9, 3, 10, 16, 37, 4, 11, 17, 38, 22, 43, 58, 93, 5, 12, 18, 39, 23, 44, 59, 94, 27, 48, 63, 98, 73, 108, 128, 163, 6, 13, 19, 40, 24, 45, 60, 95, 28, 49, 64, 99, 74, 109, 129, 164, 31, 52, 67, 102, 77, 112, 132, 167, 83, 118, 138, 173, 148, 183, 198, 219, 7, 14, 20, 41, 25, 46, 61, 96, 29, 50, 65, 100, 75, 110, 130, 165, 32, 53, 68, 103, 78, 113, 133, 168, 84, 119, 139, 174, 149, 184, 199, 220, 34, 55, 70, 105, 80, 115, 135, 170, 86, 121, 141, 176, 151, 186, 201, 222, 89, 124, 144, 179, 154, 189, 204, 225, 158, 193, 208, 229, 213, 234, 240, 247, 8, 15, 21, 42, 26, 47, 62, 97, 30, 51, 66, 101, 76, 111, 131, 166, 33, 54, 69, 104, 79, 114, 134, 169, 85, 120, 140, 175, 150, 185, 200, 221, 35, 56, 71, 106, 81, 116, 136, 171, 87, 122, 142, 177, 152, 187, 202, 223, 90, 125, 145, 180, 155, 190, 205, 226, 159, 194, 209, 230, 214, 235, 241, 248, 36, 57, 72, 107, 82, 117, 137, 172, 88, 123, 143, 178, 153, 188, 203, 224, 91, 126, 146, 181, 156, 191, 206, 227, 160, 195, 210, 231, 215, 236, 242, 249, 92, 127, 147, 182, 157, 192, 207, 228, 161, 196, 211, 232, 216, 237, 243, 250, 162, 197, 212, 233, 217, 238, 244, 251, 218, 239, 245, 252, 246, 253, 254, 255};
__constant__ int P_wieght[256] = {0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8};

// define device functions for working with boards

struct BoardDevice {
    int ones_num;  // total number of ones available to the grid
    int ones_left;  // number of ones available to place in the grid
    int last_num;  // highest number currently placed in the grid
};

__device__ __forceinline__ uint32_t get_elem(uint32_t* b_arr, int b_idx, int num_b, int elem_idx) {
    return b_arr[elem_idx * num_b + b_idx];
}

__device__ __forceinline__ void set_elem(uint32_t* b_arr, int b_idx, int num_b, int elem_idx, uint32_t elem) {
    b_arr[elem_idx * num_b + b_idx] = elem;
}

// __device__ __forceinline__ Board get_board_from_flat(uint32_t* b_arr, int b_idx, int num_b, uint32_t* packed_array) {
//     Board new_board;
//     new_board.ones_num = get_elem(b_arr, b_idx, num_b, 0);
//     new_board.ones_left = get_elem(b_arr, b_idx, num_b, 1);
//     new_board.last_num = get_elem(b_arr, b_idx, num_b, 2);
//     #pragma unroll
//     for (int i=0; i<MAX_HEIGHT; i++) {
//         packed_array[i] = get_elem(b_arr, b_idx, num_b, 3+i);
//     }
//     return new_board;
// }

__device__ void set_board_in_flat(uint32_t* b_arr, int b_idx, int num_b, const BoardDevice* B, uint32_t* packed_array) {
    set_elem(b_arr, b_idx, num_b, 0, B->ones_num);
    set_elem(b_arr, b_idx, num_b, 1, B->ones_left);
    set_elem(b_arr, b_idx, num_b, 2, B->last_num);
    #pragma unroll
    for (int i=0; i<MAX_HEIGHT; i++) {
        set_elem(b_arr, b_idx, num_b, 3+i, packed_array[i]);
    }
}

__device__ __forceinline__ int get_info(BoardDevice* B, int arr_idx, uint32_t* packed_array) {
    return unpack_info(packed_array[arr_idx]);
}

__device__ __forceinline__ void get_pos(BoardDevice* B, int arr_idx, int* x, int* y, uint32_t* packed_array) {
    uint32_t packed_int = packed_array[arr_idx];
    *x = unpack_pos_x(packed_int);
    *y = unpack_pos_y(packed_int);
}

__device__ __forceinline__ void get_pos_info(BoardDevice* B, int arr_idx, int* x, int* y, int* info, uint32_t* packed_array) {
    uint32_t packed_int = packed_array[arr_idx];
    *x = unpack_pos_x(packed_int);
    *y = unpack_pos_y(packed_int);
    *info = unpack_info(packed_int);
}

// these are ports(ish) of the functions in Board.cu
__device__ __forceinline__ void dvc_insert_element(BoardDevice* B, int x, int y, int ones_added, uint32_t* packed_array) {

    // insert an element onto our board
    packed_array[B->last_num] = pack(x, y, ones_added, 0);
    B->last_num += 1;
}

__device__ __forceinline__ void dvc_insert_one(BoardDevice* B, int x, int y, uint32_t* packed_array) {

    // insert a one onto our board
    packed_array[MAX_HEIGHT-B->ones_left] = pack(x, y, 0, 0);
    B->ones_left -= 1;
}

__device__ void dvc_remove_element(BoardDevice* B, int* one_positions, uint32_t* packed_array) {

    // remove an element from our board
    int x, y, otr;
    get_pos_info(B, B->last_num - 1, &x, &y, &otr, packed_array);
    *one_positions = 0;
    for (int i=MAX_HEIGHT-B->ones_left-otr; i<MAX_HEIGHT-B->ones_left; i++) {
        int xi, yi;
        get_pos(B, i, &xi, &yi, packed_array);
        *one_positions |= (1 << AR_IDX(xi - x, yi - y));
    }
    B->ones_left += otr;
    B->last_num -= 1;
}

__device__ int dvc_get_neighbor(BoardDevice* B, int x, int y, uint32_t* packed_array) {

    // get the lowest index neighbor of the given position
    int pos_x, pos_y;
    for (int i=0; i<B->last_num; i++) {
        get_pos(B, i, &pos_x, &pos_y, packed_array);
        if ((pos_x == x) && (pos_y == y)) {
            continue;
        }
        if ((x-1 <= pos_x) && (pos_x <= x+1)) {
            if ((y-1 <= pos_y) && (pos_y <= y+1)) {
                return i;
            }
        }
    }
    for (int i=MAX_HEIGHT-B->ones_num; i<MAX_HEIGHT-B->ones_left; i++) {
        get_pos(B, i, &pos_x, &pos_y, packed_array);
        if ((pos_x == x) && (pos_y == y)) {
            continue;
        }
        if ((x-1 <= pos_x) && (pos_x <= x+1)) {
            if ((y-1 <= pos_y) && (pos_y <= y+1)) {
                return i;
            }
        }
    }
    return MAX_HEIGHT;
}

__device__ int dvc_get_sum(BoardDevice* B, int x, int y, int target, int* least_neighbor, int* open_neighbors, uint32_t* packed_array) {

    // get the sum of elements surrounding position
    // returns -1 if position already populated
    int sum = 0;
    int pos_x, pos_y;
    *least_neighbor = MAX_HEIGHT;
    *open_neighbors = 255;
    for (int i=0; i<B->last_num; i++) {
        get_pos(B, i, &pos_x, &pos_y, packed_array);
        if ((pos_x == x) && (pos_y == y)) {
            return -1;
        }
        if ((x-1 <= pos_x) && (pos_x <= x+1)) {
            if ((y-1 <= pos_y) && (pos_y <= y+1)) {
                sum += i+2;
                *least_neighbor = MIN(*least_neighbor, i);
                *open_neighbors ^= (1 << AR_IDX(pos_x - x, pos_y - y));
                if (sum > target) return sum;
            }
        }
    }
    for (int i=MAX_HEIGHT-B->ones_num; i<MAX_HEIGHT-B->ones_left; i++) {
        get_pos(B, i, &pos_x, &pos_y, packed_array);
        if ((pos_x == x) && (pos_y == y)) {
            return -1;
        }
        if ((x-1 <= pos_x) && (pos_x <= x+1)) {
            if ((y-1 <= pos_y) && (pos_y <= y+1)) {
                sum += 1;
                *least_neighbor = MIN(*least_neighbor, i);
                *open_neighbors ^= (1 << AR_IDX(pos_x - x, pos_y - y));
                if (sum > target) return sum;
            }
        }
    }
    return sum;
}

__device__ bool dvc_look_around(BoardDevice* B, int index, int start_H, int start_P, uint32_t* packed_array) {

    // check around the location of a particular index for a spot to place the next element
    int new_x, new_y;
    int cur_sum, min_nb, open_nb, ones_needed;
    int pos_x, pos_y;
    bool vop;  // valid one positions
    get_pos(B, index, &pos_x, &pos_y, packed_array);
    for (int H=start_H; H<8; H++) {  // iterate over all spots around (i+2)
        new_x = pos_x + x_around[H];
        new_y = pos_y + y_around[H];
        cur_sum = dvc_get_sum(B, new_x, new_y, B->last_num + 2, &min_nb, &open_nb, packed_array);
        if (min_nb == index) {  // don't go in spots we've already checked
            if (cur_sum <= B->last_num + 2) {
                ones_needed = B->last_num + 2 - cur_sum;
                if (ones_needed <= MIN(B->ones_left, P_wieght[P_idxs[open_nb]])) {
                    for (int P=MAX(start_P, P_rngs[ones_needed]); P<P_rngs[ones_needed+1]; P++) {
                        if ((P_bits[P] & open_nb) != P_bits[P]) continue;  // one positions must be open
                        // validate that the one positions aren't adjacent to other numbers
                        vop = true;
                        for (int b=0; b<8; b++) {
                            if ((P_bits[P] & (1<<b)) &&
                                (dvc_get_neighbor(B, new_x+x_around[b], new_y+y_around[b], packed_array) < B->last_num)) {
                                vop = false;
                                break;
                            }
                        }
                        if (!vop) continue;
                        dvc_insert_element(B, new_x, new_y, ones_needed, packed_array);
                        for (int b=0; b<8; b++) {
                            if (P_bits[P] & (1<<b)) dvc_insert_one(B, new_x+x_around[b], new_y+y_around[b], packed_array);
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

__device__ void dvc_next_board_state(BoardDevice* B, uint32_t* packed_array) {

    // iterate a board to its next state i.e. the next position in the search
    // this function assumes that "2" has already been placed

    // first try to add the next number
    for (int i=0; i<B->last_num / 2; i++) {  // choose a num (i+2) to try to place around
        if (dvc_look_around(B, i, 0, 0, packed_array)) return;
    }
    for (int i=B->last_num - B->ones_num; i<B->last_num; i++) {  // we need to also look around high numbers
        if (dvc_look_around(B, i, 0, 0, packed_array)) return;
    }

    // failing to add a number, we'll attempt to move the current highest to a new position
    // continuing to remove elements until we succeed at moving one
    int old_x, old_y;
    int nb_x, nb_y;
    int old_nb, last_H, last_P;
    while (B->last_num - 1) {  // abort if "3" is removed
        // first find where we left off
        get_pos(B, B->last_num - 1, &old_x, &old_y, packed_array);
        old_nb = dvc_get_neighbor(B, old_x, old_y, packed_array);
        get_pos(B, old_nb, &nb_x, &nb_y, packed_array);
        last_H = AR_IDX(old_x - nb_x, old_y - nb_y);
        // remove the element and search for new spot
        dvc_remove_element(B, &last_P, packed_array);
        // start with element it was already around
        if (dvc_look_around(B, old_nb, last_H, P_idxs[last_P] + 1, packed_array)) return;
        for (int i=old_nb+1; i<B->last_num / 2; i++) {  // choose a num (i+2) to try to place around
            if (dvc_look_around(B, i, 0, 0, packed_array)) return;
        }
        for (int i=MAX(old_nb+1, B->last_num - B->ones_num); i<B->last_num; i++) {  // we need to also look around high numbers
            if (dvc_look_around(B, i, 0, 0, packed_array)) return;
        }
        // we can forgo looking around ones so long as our initial boards already have min(N,8) placed
        // if (B->last_num+2 <= 8) { // we need to also look around ones for small numbers
        //     for (int i=MAX(old_nb+1, MAX_HEIGHT - B->ones_num); i<MAX_HEIGHT-B->ones_left; i++) { 
        //         if (dvc_look_around(B, i, 0, 0)) return;
        //     }
        // }
    }
}

// Main kernel for parallel board search
__global__ void SearchKernel(uint32_t* b_arr, int* cur_max, int num_b) {
    // assign each thread one board to completely search the state space of
    int b_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (b_idx < num_b) {

        uint32_t packed_array[MAX_HEIGHT];
        int ones_num, ones_left, last_num;

        ones_num = get_elem(b_arr, b_idx, num_b, 0);
        ones_left = get_elem(b_arr, b_idx, num_b, 1);
        last_num = get_elem(b_arr, b_idx, num_b, 2);
        
        #pragma unroll
        for (int i = 0; i < MAX_HEIGHT; i++) {
            packed_array[i] = get_elem(b_arr, b_idx, num_b, 3 + i);
        }

        BoardDevice B;
        B.ones_num = ones_num;
        B.ones_left = ones_left;
        B.last_num = last_num;

        int anchor = B.last_num - 1;
        int anchor_x, anchor_y;
        get_pos(&B, anchor, &anchor_x, &anchor_y, packed_array);
        while (true) {
            dvc_next_board_state(&B, packed_array);
            if (*cur_max < B.last_num) {
                atomicMax(cur_max, B.last_num);
                set_board_in_flat(b_arr, b_idx, num_b, &B, packed_array);
            }
            int test_x, test_y;
            get_pos(&B, anchor, &test_x, &test_y, packed_array);
            if ((anchor_x != test_x) || (anchor_y != test_y)) {
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
    uint32_t* Boards_device;
    cudaMalloc((void**) &Boards_device, num_b * (MAX_HEIGHT + 3) * sizeof(uint32_t));
    cudaMemcpy(Boards_device, Boards_host, num_b * (MAX_HEIGHT+3) * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Setup the execution configuration
	dim3 blockDim(BLOCK_SIZE, 1, 1);
	dim3 gridDim((num_b + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);

    // Launch the device computation threads
    printf("launching kernel...\n");
	SearchKernel<<<gridDim,blockDim>>>(Boards_device, cur_max, num_b);
	cudaDeviceSynchronize();

    // copy results back to host
    int* host_cur_max = (int*) malloc(sizeof(int));
    cudaMemcpy(host_cur_max, cur_max, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("device max last pos: %d\n", *host_cur_max);
    cudaMemcpy(Boards_host, Boards_device, num_b * (MAX_HEIGHT+3) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    unflatten_board_list(Boards, Boards_host, num_b);
    for (int b=0; b<num_b; b++) {
        if (Boards[b].last_num == *host_cur_max) {
            CopyHostBoard(max_board, &Boards[b]);
        }
    }

    // free memory
    free(Boards_host);
    cudaFree(cur_max);
    cudaFree(Boards_device);
}
