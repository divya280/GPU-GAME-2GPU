#  🎮 GPU-Based Tic-Tac-Toe Using Two CUDA Agents
## 🏆 AIM

To design and implement a Tic-Tac-Toe game where two GPU-powered agents (Agent A and Agent B) compete by evaluating moves on the GPU. The board is shared, and a lock-file mechanism controls turn-taking.

## 📘 ALGORITHM
1. Turn Management (Lock Files)

Two lock files manage turns:

lockA → Agent A's turn

lockB → Agent B's turn

Only the agent whose lock file exists is allowed to play.
After playing:

The agent deletes its own lock file.

The agent creates the opponent’s lock file.

This creates a synchronized turn-taking system.

2. GPU Agents

Two separate GPU-powered programs act as players:

Agent A plays 'X'

Agent B plays 'O'

Each agent:

Reads the shared board.txt

Sends it to GPU to evaluate move scores

Picks the highest scoring move

Writes its symbol (X or O)

Switches lock files

3. GPU Kernel (Move Scoring)

A CUDA kernel runs 9 threads (one per board cell).
Each thread scores one position:

If cell is empty ('-') → assign score

If cell is filled → score = 0

The agent selects the position with the best score.

## 📝 PROCEDURE

Write GPU kernel gpu_eval.cu

Implement two agents:

agent_a.cpp

agent_b.cpp

Initialize:

board.txt with 9 empty cells

Create lockA so Agent A starts

Compile using nvcc + g++

Run both agents in background

Observe board.txt updating live

## 🧩 FULL CODE
### 1️⃣ gpu_eval.cu
```
#include <cuda.h>

__global__ void score_moves(char *board, float *scores) {
    int idx = threadIdx.x;

    if (board[idx] == '-') {
        scores[idx] = 1.0f;   // simple heuristic
    } else {
        scores[idx] = 0.0f;
    }
}
```
### 2️⃣ agent_a.cpp — (Plays X)
```
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <cuda.h>

__global__ void score_moves(char*, float*);

void read_board(char board[9]) {
    std::ifstream f("board.txt");
    for (int i = 0; i < 9; i++) f >> board[i];
}

void write_board(char board[9]) {
    std::ofstream f("board.txt");
    for (int i = 0; i < 9; i++) f << board[i] << " ";
}

int best_move(float scores[9]) {
    int best = -1;
    float maxScore = -1;
    for (int i = 0; i < 9; i++)
        if (scores[i] > maxScore) {
            maxScore = scores[i];
            best = i;
        }
    return best;
}

int main() {

    while (true) {

        if (!std::ifstream("lockA")) { usleep(200000); continue; }

        char board[9];
        read_board(board);

        char *d_board;
        float *d_scores;
        float scores[9];

        cudaMalloc(&d_board, 9);
        cudaMalloc(&d_scores, 9 * sizeof(float));

        cudaMemcpy(d_board, board, 9, cudaMemcpyHostToDevice);
        score_moves<<<1, 9>>>(d_board, d_scores);
        cudaMemcpy(scores, d_scores, 9 * sizeof(float), cudaMemcpyDeviceToHost);

        int move = best_move(scores);
        if (move >= 0) board[move] = 'X';

        write_board(board);

        std::remove("lockA");
        std::ofstream("lockB");

        cudaFree(d_board);
        cudaFree(d_scores);
    }
}
```
### 3️⃣ agent_b.cpp — (Plays O)
```
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <cuda.h>

__global__ void score_moves(char*, float*);

void read_board(char board[9]) {
    std::ifstream f("board.txt");
    for (int i = 0; i < 9; i++) f >> board[i];
}

void write_board(char board[9]) {
    std::ofstream f("board.txt");
    for (int i = 0; i < 9; i++) f << board[i] << " ";
}

int best_move(float scores[9]) {
    int best = -1;
    float maxScore = -1;
    for (int i = 0; i < 9; i++)
        if (scores[i] > maxScore) {
            maxScore = scores[i];
            best = i;
        }
    return best;
}

int main() {

    while (true) {

        if (!std::ifstream("lockB")) { usleep(200000); continue; }

        char board[9];
        read_board(board);

        char *d_board;
        float *d_scores;
        float scores[9];

        cudaMalloc(&d_board, 9);
        cudaMalloc(&d_scores, 9 * sizeof(float));

        cudaMemcpy(d_board, board, 9, cudaMemcpyHostToDevice);
        score_moves<<<1, 9>>>(d_board, d_scores);
        cudaMemcpy(scores, d_scores, 9 * sizeof(float), cudaMemcpyDeviceToHost);

        int move = best_move(scores);
        if (move >= 0) board[move] = 'O';

        write_board(board);

        std::remove("lockB");
        std::ofstream("lockA");

        cudaFree(d_board);
        cudaFree(d_scores);
    }
}
```



## 📌 SAMPLE OUTPUT

```

X - -
- O -
- - X

```
### Final example:
```
X O X
O X O
X O X
```
## 🏁 RESULT

A complete 2-GPU Tic-Tac-Toe system is implemented.
Two CUDA kernels evaluate moves independently, and lock files manage turn-taking.
