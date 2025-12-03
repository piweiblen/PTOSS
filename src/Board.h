#ifndef _BOARD_
#define _BOARD_

#include <stdint.h>

#define MAX_HEIGHT 128

// Board Structure declaration
typedef struct {
    int ones_num;  // total number of ones available to the grid
    int ones_left;  // number of ones available to place in the grid
    int last_num;  // highest number currently placed in the grid
    uint32_t packed_array[MAX_HEIGHT]; // Array of packed ints. Bits 0-12 contain pos_x, bits 13-25 contain pos_y (x and y coordinates of elements in grid), and bits 26-31 contain info (how many ones were placed alongside)
} Board;

void InitBoard(Board* B, int N);
void CopyHostBoard(Board* Bdest, const Board* Bsrc);
void flatten_board_list(uint32_t* flat_boards, Board* Boards, int num);
void unflatten_board_list(Board* Boards, uint32_t* flat_boards, int num);
void pretty_print(Board* B);
void insert_element(Board* B, int x, int y, int ones_added);
void insert_one(Board* B, int x, int y);
void next_board_state(Board* B);
bool equivalent(Board* B1, Board* B2);
void remove_duplicates(Board** boards, int* num_b);
void gen_all_next_boards(Board** boards, int* num_b);

// Pack a pos_x, pos_y, and info int into a single uint32_t
__host__ __device__ inline uint32_t pack(int pos_x, int pos_y, int info) {
    return ((uint32_t) pos_x & 0x1FFF) // Pack pos_x into bits 0-12
            | (((uint32_t) pos_y & 0x1FFF) << 13) // Pack pos_y into bits 13-25
            | ((((uint32_t) info & 0x3F) << 26)); // Pack info into bits 26-31
}

// Pack pos_x in packed int (without changing pos_y or info)
__host__ __device__ inline uint32_t pack_pos_x(uint32_t packed_int, int pos_x) {
    packed_int &= ~0x1FFF; // Clear old pos_x
    packed_int |= (uint32_t) pos_x & 0x1FFF; // Add new pos_x
    return packed_int;
}

// Pack pos_y in packed int (without changing pos_x or info)
__host__ __device__ inline uint32_t pack_pos_y(uint32_t packed_int, int pos_y) {
    packed_int &= ~(0x1FFF << 13); // Clear old pos_y
    packed_int |= ((uint32_t) pos_y & 0x1FFF) << 13; // Add new pos_y
    return packed_int;
}

// Pack position in packed int (without changing info)
__host__ __device__ inline uint32_t pack_pos(uint32_t packed_int, int pos_x, int pos_y) {
    packed_int = pack_pos_x(packed_int, pos_x);
    packed_int = pack_pos_y(packed_int, pos_y);
    return packed_int;
}

// Pack info in packed int (without changing pos_x or pos_y)
__host__ __device__ inline uint32_t pack_info(uint32_t packed_int, int info) {
    packed_int &= ~(0x3F << 26); // Clear old info
    packed_int |= ((uint32_t) info & 0x3F) << 26; // Add new info
    return packed_int;
}

// Pack all three int arrays into one uint32_t array
inline void pack_arrays(uint32_t* packed_array, int* pos_x, int* pos_y, int* info) {
    // Loop and pack each corresponding 3 ints into packed_ints array
    for (int i = 0; i < MAX_HEIGHT; i++) {
        packed_array[i] = pack(pos_x[i], pos_y[i], info[i]);
    }
}

// Unpack pos_x int
__host__ __device__ inline int unpack_pos_x(uint32_t packed_int) {
    return packed_int & 0x1FFF;
}

// Unpack pos_y int
__host__ __device__ inline int unpack_pos_y(uint32_t packed_int) {
    return (packed_int >> 13) & 0x1FFF;
}

// Unpack info int
__host__ __device__ inline int unpack_info(uint32_t packed_int) {
    return (packed_int >> 26) & 0x3F;
}

#endif // _BOARD_
