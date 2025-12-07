#ifndef _BOARD_
#define _BOARD_

#include <stdint.h>

#define MAX_HEIGHT 90
#define C_OFF (1<<11)

// Board Structure declaration
typedef struct {
    int ones_num;  // total number of ones available to the grid
    int ones_down;  // number of ones available to place in the grid
    int next_idx;  // next available index into packed_array
    uint32_t packed_array[MAX_HEIGHT]; // Array of packed ints. Bits 0-11 contain x, bits 12-23 contain y (coordinates of elements in grid),
                                       // bits 24-30 contain the element's value, bit 31 contains which multiboard the element is placed in
} Board;

void InitBoard(Board* B, int N);
void CopyHostBoard(Board* Bdest, const Board* Bsrc);
void flatten_board_list(uint32_t* flat_boards, Board* Boards, int num);
void unflatten_board_list(Board* Boards, uint32_t* flat_boards, int num);
void pretty_print(Board* B);
void insert_element(Board* B, int x, int y, int ones_added);
void insert_one(Board* B, int x, int y);
void next_board_state(Board* B);
void gen_boards_to_depth(Board** boards, int* num_b, const int N, const int depth);

// Pack a pos_x, pos_y, vlue and mub into a single uint32_t
__host__ __device__ inline uint32_t pack(int pos_x, int pos_y, int info, int mub) {
    return (((uint32_t) pos_x & 0xFFF) // Pack pos_x into bits 0-11
            | (((uint32_t) pos_y & 0xFFF) << 12) // Pack pos_y into bits 12-23
            | (((uint32_t) info & 0x7F) << 24) // Pack value into bits 24-30
            | (((uint32_t) mub & 0x1) << 31)); // Pack multiboard into bit 31
}

// Pack pos_x in packed int (without changing anything else)
__host__ __device__ inline uint32_t pack_pos_x(uint32_t packed_int, int pos_x) {
    packed_int &= ~0xFFF; // Clear old pos_x
    packed_int |= (uint32_t) pos_x & 0xFFF; // Add new pos_x
    return packed_int;
}

// Pack pos_y in packed int (without changing anything else)
__host__ __device__ inline uint32_t pack_pos_y(uint32_t packed_int, int pos_y) {
    packed_int &= ~(0xFFF << 12); // Clear old pos_y
    packed_int |= ((uint32_t) pos_y & 0xFFF) << 12; // Add new pos_y
    return packed_int;
}

// Pack position in packed int (without changing anything else)
__host__ __device__ inline uint32_t pack_pos(uint32_t packed_int, int pos_x, int pos_y) {
    packed_int = pack_pos_x(packed_int, pos_x);
    packed_int = pack_pos_y(packed_int, pos_y);
    return packed_int;
}

// Pack value in packed int (without changing anything else)
__host__ __device__ inline uint32_t pack_value(uint32_t packed_int, int value) {
    packed_int &= ~(0x7F << 24); // Clear old value
    packed_int |= ((uint32_t) value & 0x7F) << 24; // Add new value
    return packed_int;
}

// Pack mub in packed int (without changing anything else)
__host__ __device__ inline uint32_t pack_mub(uint32_t packed_int, int mub) {
    packed_int &= ~(0x1 << 31); // Clear old mub
    packed_int |= ((uint32_t) mub & 0x1) << 31; // Add new mub
    return packed_int;
}

// Unpack pos_x int
__host__ __device__ inline int unpack_pos_x(uint32_t packed_int) {
    return packed_int & 0xFFF;
}

// Unpack pos_y int
__host__ __device__ inline int unpack_pos_y(uint32_t packed_int) {
    return (packed_int >> 12) & 0xFFF;
}

// Unpack info int
__host__ __device__ inline int unpack_value(uint32_t packed_int) {
    return (packed_int >> 24) & 0x7F;
}

// Unpack multiboard int
__host__ __device__ inline int unpack_mub(uint32_t packed_int) {
    return (packed_int >> 31) & 0x1;
}

#endif // _BOARD_
