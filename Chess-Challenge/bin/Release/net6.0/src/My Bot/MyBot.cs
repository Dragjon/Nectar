using System;
using System.Collections;
using System.Linq;
using System.IO;
using ChessChallenge.API;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

public class MyBot : IChessBot
{

    static readonly int inputLayerSize = 384;
    static readonly int hiddenLayerSize = 32;
    static readonly int scale = 400;

    static float[,] FeatureWeights = new float[inputLayerSize, hiddenLayerSize];
    static float[] FeatureBias = new float[hiddenLayerSize];
    static float[] OutputWeights = new float[hiddenLayerSize];
    static float OutputBias;

    static float[,] BFeatureWeights = new float[inputLayerSize, hiddenLayerSize];
    static float[] BFeatureBias = new float[hiddenLayerSize];
    static float[] BOutputWeights = new float[hiddenLayerSize];
    static float BOutputBias;

    public static int SetWeights()
    {
        byte[] iweightsbytes = File.ReadAllBytes("./screlu_iweights.bin");
        byte[] ibiasesbytes = File.ReadAllBytes("./screlu_ibiases.bin");
        byte[] oweightsbytes = File.ReadAllBytes("./screlu_oweights.bin");
        byte[] obiasesbytes = File.ReadAllBytes("./screlu_obias.bin");

        byte[] biweightsbytes = File.ReadAllBytes("./screlu_biweights.bin");
        byte[] bibiasesbytes = File.ReadAllBytes("./screlu_bibiases.bin");
        byte[] boweightsbytes = File.ReadAllBytes("./screlu_boweights.bin");
        byte[] bobiasesbytes = File.ReadAllBytes("./screlu_bobias.bin");

        int row = 0;
        int col = 0;

        // Input weights
        for (int i = 0; i < inputLayerSize * hiddenLayerSize * 4; i += 4)
        {
            byte[] tmp1 = new byte[] { iweightsbytes[i], iweightsbytes[i + 1], iweightsbytes[i + 2], iweightsbytes[i + 3] };
            byte[] btmp1 = new byte[] { biweightsbytes[i], biweightsbytes[i + 1], biweightsbytes[i + 2], biweightsbytes[i + 3] };
            FeatureWeights[row, col] = BitConverter.ToSingle(tmp1, 0);
            BFeatureWeights[row, col] = BitConverter.ToSingle(btmp1, 0);

            col++;
            if (col == hiddenLayerSize)
            {
                col = 0;
                row++;
            }
        }

        // Input biases
        col = 0;
        for (int i = 0; i < hiddenLayerSize * 4; i += 4)
        {
            byte[] tmp2 = new byte[] { ibiasesbytes[i], ibiasesbytes[i + 1], ibiasesbytes[i + 2], ibiasesbytes[i + 3] };
            byte[] btmp2 = new byte[] { bibiasesbytes[i], bibiasesbytes[i + 1], bibiasesbytes[i + 2], bibiasesbytes[i + 3] };
            FeatureBias[col] = BitConverter.ToSingle(tmp2, 0);
            BFeatureBias[col] = BitConverter.ToSingle(btmp2, 0);
            col++;
        }

        // Output weights
        col = 0;
        for (int i = 0; i < hiddenLayerSize * 4; i += 4)
        {
            byte[] tmp3 = new byte[] { oweightsbytes[i], oweightsbytes[i + 1], oweightsbytes[i + 2], oweightsbytes[i + 3] };
            byte[] btmp3 = new byte[] { boweightsbytes[i], boweightsbytes[i + 1], boweightsbytes[i + 2], boweightsbytes[i + 3] };
            OutputWeights[col] = BitConverter.ToSingle(tmp3, 0);
            BOutputWeights[col] = BitConverter.ToSingle(btmp3, 0);
            col++;
        }

        // Output bias
        byte[] tmp4 = new byte[] { obiasesbytes[0], obiasesbytes[1], obiasesbytes[2], obiasesbytes[3] };
        byte[] btmp4 = new byte[] { bobiasesbytes[0], bobiasesbytes[1], bobiasesbytes[2], bobiasesbytes[3] };
        OutputBias = BitConverter.ToSingle(tmp4, 0);
        BOutputBias = BitConverter.ToSingle(btmp4, 0);

        return 0;
    }

    int initialise = SetWeights();

    private static readonly Dictionary<char, int> PieceToIndex = new Dictionary<char, int>
    {
        {'P', 0}, {'p', 0},  // Pawns
        {'N', 64}, {'n', 64}, // Knights
        {'B', 128}, {'b', 128}, // Bishops
        {'R', 192}, {'r', 192}, // Rooks
        {'Q', 256}, {'q', 256}, // Queens
        {'K', 320}, {'k', 320}  // Kings
    };

    public static float[] Encode(string fen)
    {
        // Initialize the 384-element array
        float[] boardArray = new float[384];

        // Split the FEN string to get the board layout
        string[] parts = fen.Split(' ');
        string board = parts[0];

        // Split the board part into rows
        string[] rows = board.Split('/');

        for (int rowIdx = 0; rowIdx < rows.Length; rowIdx++)
        {
            string row = rows[rowIdx];
            int colIdx = 0;

            foreach (char character in row)
            {
                if (char.IsDigit(character))
                {
                    // Empty squares, advance the column index
                    colIdx += int.Parse(character.ToString());
                }
                else
                {
                    // Piece, determine its position in the 384-element array
                    int pieceIndex = PieceToIndex[character];
                    int boardPosition = rowIdx * 8 + colIdx;
                    if (char.IsUpper(character))
                    {
                        // White piece
                        boardArray[pieceIndex + boardPosition] = 1;
                    }
                    else
                    {
                        // Black piece
                        boardArray[pieceIndex + boardPosition] = -1;
                    }
                    colIdx++;
                }
            }
        }

        return boardArray;
    }

    public class NeuralNetwork
    {

        // Sigmoid activation function
        private static float Sigmoid(float x)
        {
            return 1 / (1 + (float)Math.Exp(-x));
        }

        // SCReLU activation function
        private static float SCReLU(float x)
        {
            float clipped = Math.Clamp(x, 0, 1);
            return clipped * clipped;
        }

        public static float Predict(float[] inputs, float[,] inputWeights, float[] inputBiases, float[] outputWeights, float outputBias)
        {

            // Compute hidden layer activations
            float[] hiddenLayer = new float[32];
            for (int j = 0; j < 32; j++)
            {
                float sum = 0;
                for (int i = 0; i < 384; i++)
                {
                    sum += inputs[i] * inputWeights[i, j];
                }
                hiddenLayer[j] = SCReLU(sum + inputBiases[j]);
            }

            // Compute output layer activation
            float output = 0;
            for (int j = 0; j < 32; j++)
            {
                output += hiddenLayer[j] * outputWeights[j];
            }
            output = output + outputBias;

            return output * scale;
        }

    }

    public static int Evaluate(Board board)
    {
        float[] encoded = Encode(board.GetFenString());
        float prediction = 0;
        if (board.IsWhiteToMove)
        {
            prediction = NeuralNetwork.Predict(encoded, FeatureWeights, FeatureBias, OutputWeights, OutputBias);
        }
        else
        {
            prediction = -NeuralNetwork.Predict(encoded, BFeatureWeights, BFeatureBias, BOutputWeights, BOutputBias);
        }

        return (int)prediction;
    }


    static readonly double ttSlotSizeMB = 0.000024;
    static int hashSizeMB = 201;
    static int hashSize = Convert.ToInt32(hashSizeMB / ttSlotSizeMB);

    // this tuple is 24 bytes
    static (
        ulong, // hash 8 bytes
        ushort, // moveRaw 4 bytes
        int, // score 4 bytes
        int, // depth 4 bytes
        int // bound BOUND_EXACT=[1, 2147483647], BOUND_LOWER=2147483647, BOUND_UPPER=0 4 bytes
    )[] transpositionTable = new (ulong, ushort, int, int, int)[hashSize];

    // Variables for search
    static int rfpMargin = 55;
    static int futilityMargin = 116;
    static int mateScore = -20000;
    static int aspDepth = 3;
    static int aspDelta = 10;
    static int infinity = 30000;
    static int hardBoundTimeRatio = 10;
    static int softBoundTimeRatio = 40;


    enum ScoreType { upperbound, lowerbound, none };

    public Move Think(Board board, Timer timer)
    {
        int selDepth = 0;

        // Killer moves, 1 for each depth
        Move[] killers = new Move[4096];

        // History moves
        int[] history = new int[4096];

        int globalDepth = 1; // To be incremented for each iterative loop
        ulong nodes = 0; // To keep track of searched positions in 1 iterative loop
        Move rootBestMove = Move.NullMove;

        void printInfo(int score, ScoreType scoreType)
        {
            string scoreTypeStr = scoreType == ScoreType.upperbound ? "upperbound " : scoreType == ScoreType.lowerbound ? "lowerbound " : "";

            bool isMateScore = score < mateScore + 100 || score > -mateScore - 100;

            if (!isMateScore)
            {
                Console.WriteLine($"info depth {globalDepth} seldepth {selDepth} time {timer.MillisecondsElapsedThisTurn} nodes {nodes} nps {Convert.ToInt32(1000 * nodes / ((ulong)timer.MillisecondsElapsedThisTurn + 0.001))} hashfull {1000 * nodes / (ulong)hashSize} score cp {score} {scoreTypeStr}pv {ChessChallenge.Chess.MoveUtility.GetMoveNameUCI(new(rootBestMove.RawValue))}");
            }
            else
            {
                bool sideIsMated = score < 0;
                int mateIn;
                if (sideIsMated)
                {
                    mateIn = (mateScore + score) / 2;
                }
                else
                {
                    mateIn = (-mateScore - score) / 2;
                }
                Console.WriteLine($"info depth {globalDepth} seldepth {selDepth} time {timer.MillisecondsElapsedThisTurn} nodes {nodes} nps {Convert.ToInt32(1000 * nodes / ((ulong)timer.MillisecondsElapsedThisTurn + 0.001))} hashfull {1000 * nodes / (ulong)hashSize} score mate {mateIn} {scoreTypeStr}pv {ChessChallenge.Chess.MoveUtility.GetMoveNameUCI(new(rootBestMove.RawValue))}");
            }
        }



        // Quiescence Search
        int qSearch(int alpha, int beta, int ply)
        {
            // Hard bound time management
            if (globalDepth > 1 && timer.MillisecondsElapsedThisTurn >= timer.MillisecondsRemaining / hardBoundTimeRatio) throw null;

            selDepth = Math.Max(ply, selDepth);

            int mating_value = -mateScore - ply;

            if (mating_value < beta)
            {
                beta = mating_value;
                if (alpha >= mating_value) return mating_value;
            }

            mating_value = mateScore + ply;

            if (mating_value > alpha)
            {
                alpha = mating_value;
                if (beta <= mating_value) return mating_value;
            }

            int standPat = Evaluate(board);

            int bestScore = standPat;

            // Terminal nodes
            if (board.IsInCheckmate())
                return mateScore + ply;
            if (board.IsDraw())
                return 0;

            ref var tt = ref transpositionTable[board.ZobristKey & (ulong)(hashSize - 1)];
            var (ttHash, ttMoveRaw, score, ttDepth, ttBound) = tt;

            bool ttHit = ttHash == board.ZobristKey;
            int oldAlpha = alpha;

            if (ttHit)
            {
                if (ttBound switch
                {
                    2147483647 /* BOUND_LOWER */ => score >= beta,
                    0 /* BOUND_UPPER */ => score <= alpha,
                    // exact cutoffs at pv nodes causes problems, but need it in qsearch for matefinding
                    _ /* BOUND_EXACT */ => true,
                })
                    return score;
            }

            // Standing Pat Pruning
            if (standPat >= beta)
                return standPat;

            if (alpha < standPat)
                alpha = standPat;

            Move bestMove = Move.NullMove;

            // TT + MVV-LVA ordering
            foreach (Move move in board.GetLegalMoves(true).OrderByDescending(move => ttHit && move.RawValue == ttMoveRaw ? 9_000_000_000_000_000_000
                                          : 1_000_000_000_000_000_000 * (long)move.CapturePieceType - (long)move.MovePieceType))
            {
                if (standPat + (0b1_0100110100_1011001110_0110111010_0110000110_0010110100_0000000000 >> (int)move.CapturePieceType * 10 & 0b1_11111_11111) <= alpha)
                    break;
                nodes++;
                board.MakeMove(move);
                score = -qSearch(-beta, -alpha, ply + 1);
                board.UndoMove(move);

                if (score > alpha)
                    alpha = score;

                if (score > bestScore)
                {
                    bestMove = move;
                    bestScore = score;
                }

                // A/B pruning
                if (score >= beta)
                    break;

            }

            tt = (
            board.ZobristKey,
                    alpha > oldAlpha // don't update best move if upper bound
                    ? bestMove.RawValue
                    : ttMoveRaw,
                    Math.Clamp(bestScore, -20000, 20000),
                    0,
                    bestScore >= beta
                    ? 2147483647 /* BOUND_LOWER */
                    : alpha - oldAlpha /* BOUND_UPPER if alpha == oldAlpha else BOUND_EXACT */
            );

            return bestScore;
        }

        // Fail-Soft Negamax Search
        int search(int depth, int ply, int alpha, int beta)
        {

            // Hard time limit
            if (depth > 1 && timer.MillisecondsElapsedThisTurn >= timer.MillisecondsRemaining / hardBoundTimeRatio) throw null;

            selDepth = Math.Max(ply, selDepth);

            int mating_value = -mateScore - ply;

            if (mating_value < beta)
            {
                beta = mating_value;
                if (alpha >= mating_value) return mating_value;
            }

            mating_value = mateScore + ply;

            if (mating_value > alpha)
            {
                alpha = mating_value;
                if (beta <= mating_value) return mating_value;
            }

            bool isRoot = ply == 0;
            bool nonPv = alpha + 1 >= beta;

            ref var tt = ref transpositionTable[board.ZobristKey & (ulong)(hashSize - 1)];
            var (ttHash, ttMoveRaw, score, ttDepth, ttBound) = tt;

            bool ttHit = ttHash == board.ZobristKey;
            int oldAlpha = alpha;

            // Terminal nodes
            if (board.IsInCheckmate() && !isRoot)
                return mateScore + ply;
            if (board.IsDraw() && !isRoot)
                return 0;

            if (nonPv && ttHit)
            {
                if (ttDepth >= depth && ttBound switch
                {
                    2147483647 /* BOUND_LOWER */ => score >= beta,
                    0 /* BOUND_UPPER */ => score <= alpha,
                    // exact cutoffs at pv nodes causes problems, but need it in qsearch for matefinding
                    _ /* BOUND_EXACT */ => true,
                })
                    return score;
            }
            else if (nonPv && depth > 3)
                // Internal iterative reduction
                depth--;

            // Start Quiescence Search
            if (depth < 1)
                return qSearch(alpha, beta, ply);

            // Static eval needed for RFP and NMP
            int eval = Evaluate(board);

            // Index for killers
            int killerIndex = ply & 4095;

            // Reverse futility pruning
            if (nonPv && eval - rfpMargin * depth >= beta && !board.IsInCheck()) return eval;

            // Null move pruning
            if (nonPv && eval >= beta && board.TrySkipTurn())
            {
                eval = -search(depth - 4, ply + 1, 1 - beta, -beta);
                board.UndoSkipTurn();


                // IMPROVEMENT : RETURN BETA INSTEAD OF SCORE

                if (eval >= beta) return eval;
            }

            int bestScore = -30000;
            int moveCount = 0;

            // orderVariable(priority)
            // TT(0),  MVV-LVA ordering(1),  Killer Moves(2)

            Move bestMove = Move.NullMove;
            Move[] legals = board.GetLegalMoves();
            foreach (Move move in legals.OrderByDescending(move => ttHit && move.RawValue == ttMoveRaw ? 9_000_000_000_000_000_000
                                          : move.IsCapture ? 1_000_000_000_000_000_000 * (long)move.CapturePieceType - (long)move.MovePieceType
                                          : move == killers[killerIndex] ? 500_000_000_000_000_000
                                          : history[move.RawValue & 4095]))
            {
                moveCount++;

                nodes++;

                int reduction = moveCount > 3 && nonPv && !move.IsCapture ? 1 : 0;

                board.MakeMove(move);

                // Check extension
                int moveExtension = board.IsInCheck() ? 1 : 0;

                score = 0;


                // Principle variation search
                if (moveCount == 1)
                {
                    score = -search(depth - 1 + moveExtension, ply + 1, -beta, -alpha);
                }
                else
                {
                    score = -search(depth - 1 + moveExtension - reduction, ply + 1, -alpha - 1, -alpha);

                    if (reduction > 0 && score > alpha)
                        score = -search(depth - 1 + moveExtension, ply + 1, -alpha - 1, -alpha);

                    if (score > alpha && score < beta)
                    {
                        score = -search(depth - 1 + moveExtension, ply + 1, -beta, -alpha);
                    }
                }

                board.UndoMove(move);

                // Updating stuff
                if (score > alpha)
                    alpha = score;

                if (score > bestScore)
                {
                    bestScore = score;
                    bestMove = move;
                    if (isRoot)
                        rootBestMove = move;

                    // A/B pruning
                    if (score >= beta)
                    {

                        if (!move.IsCapture)
                        {
                            history[move.RawValue & 4095] += depth * depth;

                            // Keep track of the first killers for each ply
                            if (killers[killerIndex] == Move.NullMove)
                                killers[killerIndex] = move;
                        }
                        break;
                    }
                }


                // Extended futility pruning
                if (nonPv && depth <= 4 && !move.IsCapture && (eval + futilityMargin * depth * depth < alpha) && bestScore > mateScore + 100)
                    break;

            }

            tt = (
                    board.ZobristKey,
                    alpha > oldAlpha // don't update best move if upper bound
                    ? bestMove.RawValue
                    : ttMoveRaw,
                    Math.Clamp(bestScore, -20000, 20000),
                    depth,
                    bestScore >= beta
                    ? 2147483647 /* BOUND_LOWER */
                    : alpha - oldAlpha /* BOUND_UPPER if alpha == oldAlpha else BOUND_EXACT */
            );

            return bestScore;

        }


        try
        {

            nodes = 0;
            int score = 0;
            // Soft time limit
            for (; timer.MillisecondsElapsedThisTurn < timer.MillisecondsRemaining / softBoundTimeRatio; ++globalDepth)
            {
                int alpha = -infinity;
                int beta = infinity;
                int delta = 0;
                if (globalDepth > aspDepth)
                {
                    delta = aspDelta;
                    alpha = score - delta;
                    beta = score + delta;
                }
                killers = new Move[4096];
                int newScore;
                while (true)
                {
                    newScore = search(globalDepth, 0, alpha, beta);
                    if (newScore <= alpha)
                    {
                        beta = (newScore + beta) / 2;
                        alpha = Math.Max(newScore - delta, -infinity);

                        printInfo(alpha, ScoreType.upperbound);
                    }
                    else if (newScore >= beta)
                    {
                        beta = Math.Min(newScore + delta, infinity);

                        printInfo(beta, ScoreType.lowerbound);
                    }
                    else
                        break;
                    if (delta <= 500)
                        delta += delta;
                    else
                        delta = infinity;
                }

                score = newScore;

                printInfo(score, ScoreType.none);
            }
        }
        catch { }

        return rootBestMove;
    }
}