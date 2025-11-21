#ifndef _BOARD_
#define _BOARD_

#define MAX_HEIGHT 128  // Placeholder value, may want to make this dynamic based on N later

// Board Structure declaration
typedef struct {
    int ones_num;  // total number of ones available to the grid
    int ones_left;  // number of ones available to place in the grid
    // x and y postions of elements in our theoretical grid
    // we also store the ones at the end of these arrays
    int* pos_x;
    int* pos_y;
    int* info;  // how many ones were placed alongside
    int last_num;  // highest number currently placed in the grid
} Board;

void InitBoard(Board* B, int N);
Board InitDeviceBoard(const Board B);
void CopyHostBoard(Board* Bdest, const Board* Bsrc);
void CopyToDeviceBoard(Board* Bdevice, const Board* Bhost);
void CopyFromDeviceBoard(Board* Bhost, const Board* Bdevice);
void FreeDeviceBoard(Board* BM);
void FreeBoard(Board* B);
void pretty_print(Board* B);
void insert_element(Board* B, int x, int y, int ones_added);
void insert_one(Board* B, int x, int y);
void remove_element(Board* B);
void remove_one(Board* B, int x, int y);
void next_board_state(Board* B);

#endif // _BOARD_
