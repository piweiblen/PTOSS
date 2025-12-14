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
    int ones_down;  // number of ones available to place in the grid
    int next_idx;  // highest number currently placed in the grid
};

__device__ __forceinline__ uint32_t get_elem(uint32_t* b_arr, int b_idx, int num_b, int elem_idx) {
    return b_arr[elem_idx * num_b + b_idx];
}

__device__ __forceinline__ void set_elem(uint32_t* b_arr, int b_idx, int num_b, int elem_idx, uint32_t elem) {
    b_arr[elem_idx * num_b + b_idx] = elem;
}

__device__ __forceinline__ void get_board_from_flat(BoardDevice* B, uint32_t* packed_array, uint32_t* b_arr, int b_idx, int num_b) {
    B->ones_num = get_elem(b_arr, b_idx, num_b, 0);
    B->ones_down = get_elem(b_arr, b_idx, num_b, 1);
    B->next_idx = get_elem(b_arr, b_idx, num_b, 2);
    for (int i=0; i<B->next_idx; i++) {
        packed_array[i] = get_elem(b_arr, b_idx, num_b, 3+i);
    }
}

__device__ __forceinline__ void set_board_in_flat(uint32_t* b_arr, int b_idx, int num_b, const BoardDevice* B, uint32_t* packed_array) {
    set_elem(b_arr, b_idx, num_b, 0, B->ones_num);
    set_elem(b_arr, b_idx, num_b, 1, B->ones_down);
    set_elem(b_arr, b_idx, num_b, 2, B->next_idx);
    for (int i=0; i<B->next_idx; i++) {
        set_elem(b_arr, b_idx, num_b, 3+i, packed_array[i]);
    }
}

// these are ports(ish) of the functions in Board.cu
// insert an element onto our board
__device__ __forceinline__ void dvc_insert_element(BoardDevice* B, uint32_t* packed_array, int value, int x, int y, int anchor) {

    packed_array[B->next_idx] = pack(x, y, anchor, value, 0);
    B->next_idx += 1;
}

// insert a one onto our board
__device__ __forceinline__ void dvc_insert_one(BoardDevice* B, uint32_t* packed_array, int x, int y) {

    packed_array[B->next_idx] = pack(x, y, 0, 1, 0);
    B->next_idx += 1;
    B->ones_down += 1;
}

// remove an element from our board
__device__ int dvc_remove_element(BoardDevice* B, uint32_t* packed_array) {

    int x = unpack_pos_x(packed_array[(B->next_idx - 1)]);
    int y = unpack_pos_y(packed_array[(B->next_idx - 1)]);
    B->next_idx -= 1;
    int one_positions = 0;
    while (unpack_value(packed_array[(B->next_idx - 1)]) == 1) {
        int xi = unpack_pos_x(packed_array[(B->next_idx - 1)]);
        int yi = unpack_pos_y(packed_array[(B->next_idx - 1)]);
        one_positions |= (1 << AR_IDX(xi - x, yi - y));
        B->ones_down -= 1;
        B->next_idx -= 1;
    }
    return one_positions;
}

// check if a given position is clear for inserting a 1
__device__ bool dvc_clear_for_one(BoardDevice* B, uint32_t* packed_array, int x, int y) {

    for (int i=2; i<B->next_idx; i++) {  // first two are always 1s
        int pos_x = unpack_pos_x(packed_array[i]);
        int pos_y = unpack_pos_y(packed_array[i]);
        int value = unpack_value(packed_array[i]);
        if ((x-1 <= pos_x) && (pos_x <= x+1) && 
            (y-1 <= pos_y) && (pos_y <= y+1) &&
            (value > 1)) {
            return false;
        }
    }
    return true;
}

// get the sum of elements surrounding position
// returns INT_MAX if position already populated
__device__ int dvc_get_sum(BoardDevice* B, uint32_t* packed_array, int x, int y, int target, int anchor, int* open_neighbors) {

    *open_neighbors = 255;
    int sum = 0;
    for (int i=0; i<B->next_idx; i++) {
        int pos_x = unpack_pos_x(packed_array[i]);
        int pos_y = unpack_pos_y(packed_array[i]);
        if ((pos_x == x) && (pos_y == y)) {
            return INT_MAX;
        }
        if ((x-1 <= pos_x) && (pos_x <= x+1) && (y-1 <= pos_y) && (pos_y <= y+1)) {
            if (i < anchor) return INT_MAX;  // don't go in spots we've already checked
            sum += unpack_value(packed_array[i]);
            if (sum > target) return INT_MAX;
            *open_neighbors &= ~(1 << AR_IDX(pos_x - x, pos_y - y));
        }
    }
    return sum;
}

// check around the location of a particular index for a spot to place the next element
__device__ bool dvc_look_around(BoardDevice* B, uint32_t* packed_array, int index, int start_H, int start_P) {

    int cur_sum, open_nb, ones_needed;
    int target = B->next_idx - B->ones_down + 2;
    int pos_x = unpack_pos_x(packed_array[index]);
    int pos_y = unpack_pos_y(packed_array[index]);
    for (int H=start_H; H<8; H++) {  // iterate over all spots around (i+2)
        int new_x = pos_x + x_around[H];
        int new_y = pos_y + y_around[H];
        cur_sum = dvc_get_sum(B, packed_array, new_x, new_y, target, index, &open_nb);
        if (cur_sum <= target) {
            ones_needed = target - cur_sum;
            if (ones_needed <= MIN(B->ones_num - B->ones_down, P_wieght[P_idxs[open_nb]])) {
                for (int P=MAX(start_P, P_rngs[ones_needed]); P<P_rngs[ones_needed+1]; P++) {
                    if ((P_bits[P] & open_nb) != P_bits[P]) continue;  // one positions must be open
                    // validate that the one positions aren't adjacent to other numbers
                    bool vop = true;  // valid one positions
                    for (int b=0; b<8; b++) {
                        if ((P_bits[P] & (1<<b)) &&
                            (!dvc_clear_for_one(B, packed_array, new_x+x_around[b], new_y+y_around[b]))) {
                            vop = false;
                            break;
                        }
                    }
                    if (!vop) continue;
                    for (int b=0; b<8; b++) {
                        if (P_bits[P] & (1<<b)) dvc_insert_one(B, packed_array, new_x+x_around[b], new_y+y_around[b]);
                    }
                    dvc_insert_element(B, packed_array, target, new_x, new_y, index);
                    return true;
                }
            }
        }
        start_P = 0;
    }
    return false;
}

// iterate a board to its next state i.e. the next position in the search
__device__ bool dvc_next_board_state(BoardDevice* B, uint32_t* packed_array, int anchor) {

    // this function assumes that "2" has already been placed
    // first try to add the next number
    for (int i=0; i<B->next_idx; i++) {
        if (((B->next_idx - B->ones_down + 2) / 2 < unpack_value(packed_array[i])) &&
            (unpack_value(packed_array[i]) < (B->next_idx - B->ones_down + 2) - 5)) {
            continue;
        }
        if (dvc_look_around(B, packed_array, i, 0, 0)) return true;
    }

    // failing to add a number, we'll attempt to move the current highest to a new position
    // continuing to remove elements until we succeed at moving one
    while (B->next_idx - 1 > anchor) {  // abort if the anchor is removed
        // first find where we left off
        int old_x = unpack_pos_x(packed_array[B->next_idx - 1]);
        int old_y = unpack_pos_y(packed_array[B->next_idx - 1]);
        int old_nb = unpack_anchor(packed_array[B->next_idx - 1]);
        int nb_x = unpack_pos_x(packed_array[old_nb]);
        int nb_y = unpack_pos_y(packed_array[old_nb]);
        int last_H = AR_IDX(old_x - nb_x, old_y - nb_y);
        // remove the element and search for new spot
        int last_P = dvc_remove_element(B, packed_array);
        // start with element it was already around
        if (dvc_look_around(B, packed_array, old_nb, last_H, P_idxs[last_P] + 1)) return true;
        for (int i=old_nb+1; i<B->next_idx; i++) {
            if (((B->next_idx - B->ones_down + 2) / 2 < unpack_value(packed_array[i])) &&
                (unpack_value(packed_array[i]) < (B->next_idx - B->ones_down + 2) - 5)) {
                continue;
            }
            if (dvc_look_around(B, packed_array, i, 0, 0)) return true;
        }
    }
    return false;
}

// Main kernel for parallel board search
__global__ void SearchKernel(uint32_t* b_arr, int* cur_max, int num_b) {
    
    // assign each thread one board to completely search the state space of
    int b_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (b_idx < num_b) {
        BoardDevice B;
        uint32_t packed_array[MAX_HEIGHT];
        get_board_from_flat(&B, packed_array, b_arr, b_idx, num_b);
        int anchor_idx = B.next_idx - 1;
        while (dvc_next_board_state(&B, packed_array, anchor_idx)) {
            if (*cur_max < B.next_idx) {
                atomicMax(cur_max, B.next_idx);
                set_board_in_flat(b_arr, b_idx, num_b, &B, packed_array);
            }
        }
    }
}

// Kernel calling function specification
void LaunchSearchKernel(Board* Boards, uint32_t** Boards_device, int** cur_max, int num_b) {

    // do a bit of data marshaling
    uint32_t* Boards_host = (uint32_t*) malloc(num_b * (MAX_HEIGHT+3) * sizeof(uint32_t));
    flatten_board_list(Boards_host, Boards, num_b);

    // set up device memory
    cudaMalloc((void**) cur_max, sizeof(int));
	cudaMemset(*cur_max, 0, sizeof(int));
    cudaMalloc((void**) Boards_device, num_b * (MAX_HEIGHT+3) * sizeof(uint32_t));
    cudaMemcpy(*Boards_device, Boards_host, num_b * (MAX_HEIGHT+3) * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Setup the execution configuration
	dim3 blockDim(BLOCK_SIZE, 1, 1);
	dim3 gridDim((num_b + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);

    // Launch the device computation threads
    printf("launching kernel...\n");
	SearchKernel<<<gridDim,blockDim>>>(*Boards_device, *cur_max, num_b);

    // free memory
    free(Boards_host);
}

// Kernel calling function specification
Board GetKernelResult(uint32_t** Boards_device, int** cur_max, int num_b) {

    // copy results back to host
    Board max_board;
    uint32_t* Boards_host = (uint32_t*) malloc(num_b * (MAX_HEIGHT+3) * sizeof(uint32_t));
    Board* Boards = (Board *) malloc(num_b * sizeof(Board));
    int* host_cur_max = (int*) malloc(sizeof(int));
	cudaDeviceSynchronize();
    cudaMemcpy(host_cur_max, *cur_max, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(Boards_host, *Boards_device, num_b * (MAX_HEIGHT+3) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    unflatten_board_list(Boards, Boards_host, num_b);
    for (int b=0; b<num_b; b++) {
        if (Boards[b].next_idx == *host_cur_max) {
            CopyHostBoard(&max_board, &Boards[b]);
        }
    }

    // free memory
    free(Boards_host);
    cudaFree(*Boards_device);
    cudaFree(*cur_max);
    return max_board;
}
