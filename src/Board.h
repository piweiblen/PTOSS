#ifndef _BOARD_
#define _BOARD_

#include <stdint.h>

#define MAX_HEIGHT 128

// Board Structure declaration
typedef struct {
    int ones_num;  // total number of ones available to the grid
    int ones_left;  // number of ones available to place in the grid
    int last_num;  // highest number currently placed in the grid
    // x and y postions of elements in our theoretical grid
    // we also store the ones at the end of these arrays
    int pos_x[MAX_HEIGHT];
    int pos_y[MAX_HEIGHT];
    int info[MAX_HEIGHT];  // how many ones were placed alongside
                           // TODO: it may be worth eliminating the 3 arrays and "packing" them into a single array
                           // it'll make some of the code messier but save a lot of GPU memory
                           // and we definitely don't need more than, say, 6 bits for info, and 13 each for x_pos and y_pos
                           // Packing and unpacking functions have been added, but code hasn't been updated accordingly yet
} Board;

void InitBoard(Board* B, int N);
void CopyHostBoard(Board* Bdest, const Board* Bsrc);
void flatten_board_list(int* flat_boards, Board* Boards, int num);
void unflatten_board_list(Board* Boards, int* flat_boards, int num);
void pretty_print(Board* B);
void insert_element(Board* B, int x, int y, int ones_added);
void insert_one(Board* B, int x, int y);
void remove_element(Board* B);
void remove_one(Board* B, int x, int y);
void next_board_state(Board* B);
uint32_t pack_ints(int pos_x, int pos_y, int info);
uint32_t pack_pos_x(uint32_t packed_int, int pos_x);
uint32_t pack_pos_y(uint32_t packed_int, int pos_y);
uint32_t pack_info(uint32_t packed_int, int info);
uint32_t* pack_arrays(uint32_t packed_array, int* pos_x, int* pos_y, int* info);
int unpack_pos_x(uint32_t packed_int);
int unpack_pos_y(uint32_t packed_int);
int unpack_info(uint32_t packed_int);

#endif // _BOARD_
