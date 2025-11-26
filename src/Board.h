#ifndef _BOARD_
#define _BOARD_

#define MAX_HEIGHT 128  // Placeholder value, may want to make this dynamic based on N later

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

#endif // _BOARD_
