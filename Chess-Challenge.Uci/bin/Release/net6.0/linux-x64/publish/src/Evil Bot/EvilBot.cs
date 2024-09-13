using System.Linq;
using ChessChallenge.API;

namespace ChessChallenge.Example
{
    // A simple bot that can spot mate in one, and always captures the most valuable piece it can.
    // Plays randomly otherwise.
    public class EvilBot : IChessBot
    {

        // Material Values Tuned offsetted by +33
        // PESTO Mg: 0, 82, 337, 365, 477, 1025, 0
        // PESTO Eg: 0, 94, 281, 297, 512,  936,  0
        // Original average: 0, 88, 309, 331, 495, 981, 0
        // Offsetted average: 0, 121, 342, 364, 528, 1014, 0

        // A General Pesto Square Table For All Pieces Compressed offsetted by +33 to keep values positive
        /* Original
        -33, -6, -3, -4, 7, -9, 0, -10,
        13, 26, 28, 30, 31, 43, 22, 10,
        -2, 19, 22, 25, 27, 38, 32, 11,
        -6, 5, 10, 17, 17, 20, 10, -1,
        -15, 0, 4, 9, 9, 4, 4, -11,
        -15, -5, 0, 0, 3, 2, 4, -11,
        -19, -10, -5, -7, -3, 1, -2, -19,
        -26, -11, -12, -12, -7, -14, -16, -24,
         */

        /* Offesetted
         0, 27, 30, 29, 40, 24, 33, 23, 
        46, 59, 61, 63, 64, 76, 55, 43, 
        31, 52, 55, 58, 60, 71, 65, 44, 
        27, 38, 43, 50, 50, 53, 43, 32, 
        18, 33, 37, 42, 42, 37, 37, 22, 
        18, 28, 33, 33, 36, 35, 37, 22, 
        14, 23, 28, 26, 30, 34, 31, 14, 
        7, 22, 21, 21, 26, 19, 17, 9,
         */

        Move[] TT = new Move[67108864];
        public Move Think(Board board, Timer timer)
        {
            var (globalDepth, rootBestMove) = (0, Move.NullMove);

            // Negamax Search Algorithm
            int Negamax(int depth, int alpha, int beta, bool isRoot = false)
            {
                // Timeout
                if (globalDepth > 1 && timer.MillisecondsElapsedThisTurn >= timer.MillisecondsRemaining / 20)
                    throw null; // Suggested by Analog Hors in place of throw new Exception(); Because you are not allowed to throw null, which causes a null reference exception

                var (key, tScore, moves, isQSearch, packedVals, pceValues, max, notNodeIsCheck) =
                (board.ZobristKey % 67108864,
                  10,
                  0,
                  depth < 1,
                  new[] {
          1666639897670064896,
          3114041506172582702,
          3188908335155328031,
          2318004922918577691,
          1595722505998639378,
          1595720281054321682,
          1017569553491433230,
          653324423689213447
                  },
                  new[] {
          0,
          121,
          342,
          364,
          528,
          1014,
          0
                  },
                  -100000,
                  !board.IsInCheck()
                );

                foreach (var pl in board.GetAllPieceLists())
                    foreach (var p in pl)
                    {
                        var square = p.Square;
                        tScore +=
                          (board.IsWhiteToMove ? 1 : -1) *
                          (pl.IsWhitePieceList ? 1 : -1) *
                          (
                            ((int)packedVals[((square.Index) ^ (pl.IsWhitePieceList ? 56 : 0)) / 8] >> ((square.Index) & 56) & 0xFF) +
                            pceValues[(int)p.PieceType]
                          );

                    }

                // Quiscence Search Stand Pat pruning
                if (isQSearch)
                {
                    max = tScore;
                    if (tScore >= beta)
                        return beta;
                    if (alpha < tScore)
                        alpha = tScore;
                }
                else
                {
                    // Reverse Futility Pruning
                    if (notNodeIsCheck && tScore - 85 * depth >= beta)
                        return tScore;

                    // Check Extension gained +49.2 elo
                    if (!notNodeIsCheck) depth++;

                    if (TT[key] == Move.NullMove) depth--;
                }

                foreach (Move move in board.GetLegalMoves(isQSearch).OrderByDescending(move => (
                  move == TT[key],
                  move.CapturePieceType,
                  0 - move.MovePieceType)))
                {
                    board.MakeMove(move);

                    // Late Move Reduction
                    tScore = board.IsInCheckmate() ? 30000 : board.IsDraw() ? 0 : -Negamax(depth - (moves++ > 5 && !move.IsCapture && notNodeIsCheck ? 2 : 1), -beta, -alpha);

                    board.UndoMove(move);

                    if (tScore > max)
                    {
                        TT[key] = move;
                        max = tScore;

                        if (isRoot)
                            rootBestMove = move;

                        if (tScore > alpha)
                            alpha = tScore;

                        if (tScore >= beta)
                            break;
                    }

                }

                return max;
            }

            try
            {
                while (timer.MillisecondsElapsedThisTurn < timer.MillisecondsRemaining / 5)
                    Negamax(++globalDepth, -100000, 100000, true);
            }
            catch { }

            return rootBestMove;
        }

    }
}