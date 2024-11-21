using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using ChessChallenge.API;

public class MyBot : IChessBot
{

    static readonly int inputLayerSize = 768;
    static readonly int hiddenLayerSize = 16;
    static readonly int scale = 150;
    static readonly int quantise = 255;
    static readonly int quantiseSquared = 255 * 255;
    static int[] FeatureWeights = new int[inputLayerSize * hiddenLayerSize];
    static int[] FeatureBias = new int[hiddenLayerSize];
    static int[] OutputWeights = new int[hiddenLayerSize];
    static int OutputBias;

    public static int SetWeights()
    {
        byte[] iweightsbytes = File.ReadAllBytes("./screlu_iweights.bin");
        byte[] ibiasesbytes = File.ReadAllBytes("./screlu_ibiases.bin");
        byte[] oweightsbytes = File.ReadAllBytes("./screlu_oweights.bin");
        byte[] obiasesbytes = File.ReadAllBytes("./screlu_obias.bin");

        int row = 0;
        int col = 0;

        // Input weights
        for (int i = 0; i < inputLayerSize * hiddenLayerSize * 4; i += 4)
        {
            byte[] tmp1 = new byte[] { iweightsbytes[i], iweightsbytes[i + 1], iweightsbytes[i + 2], iweightsbytes[i + 3] };
            FeatureWeights[row * hiddenLayerSize + col] = (int)(BitConverter.ToSingle(tmp1, 0) * quantise);

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
            FeatureBias[col] = (int)(BitConverter.ToSingle(tmp2, 0) * quantise);
            col++;
        }

        // Output weights
        col = 0;
        for (int i = 0; i < hiddenLayerSize * 4; i += 4)
        {
            byte[] tmp3 = new byte[] { oweightsbytes[i], oweightsbytes[i + 1], oweightsbytes[i + 2], oweightsbytes[i + 3] };
            OutputWeights[col] = (int)(BitConverter.ToSingle(tmp3, 0) * quantise);
            col++;
        }

        // Output bias
        byte[] tmp4 = new byte[] { obiasesbytes[0], obiasesbytes[1], obiasesbytes[2], obiasesbytes[3] };
        OutputBias = (int)(BitConverter.ToSingle(tmp4, 0) * quantise * quantise);

        return 0;
    }

    int initialise = SetWeights();

    public static int[] Encode(string fen)
    {
        int[] boardArray = new int[inputLayerSize];
        ReadOnlySpan<char> span = fen.AsSpan();
        int spaceIndex = span.IndexOf(' ');
        ReadOnlySpan<char> boardSpan = span.Slice(0, spaceIndex);
        ReadOnlySpan<char> rows = boardSpan;

        int rowIdx = 0;
        int colIdx = 0;

        for (int i = 0; i < rows.Length; i++)
        {
            var character = rows[i];
            if (character == '/')
            {
                rowIdx++;
                colIdx = 0;
            }
            else if (char.IsDigit(character))
            {
                colIdx += character - '0';
            }
            else
            {
                int pieceIndex = character switch
                {
                    'P' => 0,
                    'N' => 64,
                    'B' => 128,
                    'R' => 192,
                    'Q' => 256,
                    'K' => 320,
                    'p' => 384,
                    'n' => 448,
                    'b' => 512,
                    'r' => 576,
                    'q' => 640,
                    'k' => 704,
                    _ => throw new InvalidOperationException("Invalid piece character")
                };

                int boardPosition = rowIdx * 8 + colIdx;

                boardArray[pieceIndex + boardPosition] = 1;

                colIdx++;
            }
        }

        return boardArray;
    }

    static int[] accumulators = new int[inputLayerSize];

    public void undoUpdateAccumulators(Piece piece, Move move)
    {
        int startSqIndex = move.StartSquare.Index ^ 56;
        int targetSqIndex = move.TargetSquare.Index ^ 56;
        int pieceIndex = (int)(piece.PieceType - 1) * 64 + (piece.IsWhite ? 0 : 384);

        accumulators[pieceIndex + startSqIndex] = 1;
        accumulators[pieceIndex + targetSqIndex] = 0;

        if (move.IsEnPassant)
        {
            accumulators[384 * (piece.IsWhite ? 1 : 0) + targetSqIndex + (piece.IsWhite ? 8 : -8)] = 1;
        }

        else if (move.IsCapture)
        {
            int capturedPieceIndex = (int)(move.CapturePieceType - 1) * 64 + (piece.IsWhite ? 384 : 0);
            accumulators[capturedPieceIndex + targetSqIndex] = 1;

            if (move.IsPromotion)
            {
                int promoPieceIndex = (int)(move.PromotionPieceType - 1) * 64 + (piece.IsWhite ? 0 : 384);
                accumulators[promoPieceIndex + targetSqIndex] = 0;
            }
        }

        else if (move.IsPromotion)
        {
            int promoPieceIndex = (int)(move.PromotionPieceType - 1) * 64 + (piece.IsWhite ? 0 : 384);
            accumulators[promoPieceIndex + targetSqIndex] = 0;
        }


        else if (move.IsCastles)
        {
            int rookOffset = piece.IsWhite ? 192 : 576;
            switch (targetSqIndex)
            {
                case 62: // White KingSide
                    accumulators[rookOffset + 63] = 1;
                    accumulators[rookOffset + 61] = 0;
                    break;
                case 58: // White QueenSide
                    accumulators[rookOffset + 56] = 1;
                    accumulators[rookOffset + 59] = 0;
                    break;
                case 6: // Black KingSide
                    accumulators[rookOffset + 7] = 1;
                    accumulators[rookOffset + 5] = 0;
                    break;
                case 2: // Black QueenSide
                    accumulators[rookOffset + 0] = 1;
                    accumulators[rookOffset + 3] = 0;
                    break;
            }
        }
    }

    public void updateAccumulators(Piece piece, Move move)
    {
        int startSqIndex = move.StartSquare.Index ^ 56;
        int targetSqIndex = move.TargetSquare.Index ^ 56;
        int pieceIndex = (int)(piece.PieceType - 1) * 64 + (piece.IsWhite ? 0 : 384);

        accumulators[pieceIndex + startSqIndex] = 0;
        accumulators[pieceIndex + targetSqIndex] = 1;


        if (move.IsEnPassant)
        {
            accumulators[384 * (piece.IsWhite ? 1 : 0) + targetSqIndex + (piece.IsWhite ? 8 : -8)] = 0;
        }

        else if (move.IsCapture)
        {
            int capturedPieceIndex = (int)(move.CapturePieceType - 1) * 64 + (piece.IsWhite ? 384 : 0);
            accumulators[capturedPieceIndex + targetSqIndex] = 0;


            if (move.IsPromotion)
            {
                int promoPieceIndex = (int)(move.PromotionPieceType - 1) * 64 + (piece.IsWhite ? 0 : 384);
                accumulators[promoPieceIndex + targetSqIndex] = 1;
                accumulators[pieceIndex + targetSqIndex] = 0;
            }

        }

        else if (move.IsPromotion)
        {
            int promoPieceIndex = (int)(move.PromotionPieceType - 1) * 64 + (piece.IsWhite ? 0 : 384);
            accumulators[promoPieceIndex + targetSqIndex] = 1;
            accumulators[pieceIndex + targetSqIndex] = 0;
        }

        else if (move.IsCastles)
        {
            int rookOffset = piece.IsWhite ? 192 : 576;
            switch (targetSqIndex)
            {
                case 62: // White KingSide
                    accumulators[rookOffset + 63] = 0;
                    accumulators[rookOffset + 61] = 1;
                    break;
                case 58: // White QueenSide
                    accumulators[rookOffset + 56] = 0;
                    accumulators[rookOffset + 59] = 1;
                    break;
                case 6: // Black KingSide
                    accumulators[rookOffset + 7] = 0;
                    accumulators[rookOffset + 5] = 1;
                    break;
                case 2: // Black QueenSide
                    accumulators[rookOffset + 0] = 0;
                    accumulators[rookOffset + 3] = 1;
                    break;
            }
        }
    }

    public class NeuralNetwork
    {
        // SCReLU activation function
        private static int SCReLU(int x)
        {
            return (x < 0) ? 0 : (x > quantise) ? quantiseSquared : x * x;
        }

        public static unsafe int Predict(int[] inputs)
        {
            int* hiddenLayer = stackalloc int[hiddenLayerSize]; // Use stack allocation for the hidden layer.

            fixed (int* pInputs = inputs)
            fixed (int* pInputWeights = FeatureWeights)
            fixed (int* pInputBiases = FeatureBias)
            fixed (int* pOutputWeights = OutputWeights)
            {
                // Process the hidden layer with loop unrolling and SIMD-like processing.
                for (int j = 0; j < hiddenLayerSize; j++)
                {
                    int sum = 0;
                    int* pWeights = pInputWeights + j;

                    int i = 0;
                    int loopEnd = inputLayerSize - (inputLayerSize % 8);

                    // SIMD-like unrolling (8x at a time)
                    for (; i < loopEnd; i += 8)
                    {
                        sum += pInputs[i] * pWeights[0] +
                               pInputs[i + 1] * pWeights[hiddenLayerSize] +
                               pInputs[i + 2] * pWeights[2 * hiddenLayerSize] +
                               pInputs[i + 3] * pWeights[3 * hiddenLayerSize] +
                               pInputs[i + 4] * pWeights[4 * hiddenLayerSize] +
                               pInputs[i + 5] * pWeights[5 * hiddenLayerSize] +
                               pInputs[i + 6] * pWeights[6 * hiddenLayerSize] +
                               pInputs[i + 7] * pWeights[7 * hiddenLayerSize];

                        pWeights += 8 * hiddenLayerSize;
                    }

                    // Handle any remaining elements
                    for (; i < inputLayerSize; i++)
                    {
                        sum += pInputs[i] * *pWeights;
                        pWeights += hiddenLayerSize;
                    }

                    hiddenLayer[j] = SCReLU(sum + pInputBiases[j]);
                }

                // Output layer computation
                int output = 0;

                for (int j = 0; j < hiddenLayerSize; j++)
                {
                    output += hiddenLayer[j] * pOutputWeights[j];
                }

                return (output / quantise + OutputBias) * scale / quantiseSquared;
            }
        }

        public static unsafe int PredictWithAcc()
        {
            int* hiddenLayer = stackalloc int[hiddenLayerSize]; // Use stack allocation for the hidden layer.

            fixed (int* pAccumulators = accumulators)
            fixed (int* pInputWeights = FeatureWeights)
            fixed (int* pInputBiases = FeatureBias)
            fixed (int* pOutputWeights = OutputWeights)
            {
                // Process the hidden layer with loop unrolling and SIMD-like processing.
                for (int j = 0; j < hiddenLayerSize; j++)
                {
                    int sum = 0;
                    int* pWeights = pInputWeights + j;

                    int i = 0;
                    int loopEnd = inputLayerSize - (inputLayerSize % 8);

                    // SIMD-like unrolling (8x at a time)
                    for (; i < loopEnd; i += 8)
                    {
                        sum += pAccumulators[i] * pWeights[0] +
                               pAccumulators[i + 1] * pWeights[hiddenLayerSize] +
                               pAccumulators[i + 2] * pWeights[2 * hiddenLayerSize] +
                               pAccumulators[i + 3] * pWeights[3 * hiddenLayerSize] +
                               pAccumulators[i + 4] * pWeights[4 * hiddenLayerSize] +
                               pAccumulators[i + 5] * pWeights[5 * hiddenLayerSize] +
                               pAccumulators[i + 6] * pWeights[6 * hiddenLayerSize] +
                               pAccumulators[i + 7] * pWeights[7 * hiddenLayerSize];

                        pWeights += 8 * hiddenLayerSize;
                    }

                    // Handle any remaining elements
                    for (; i < inputLayerSize; i++)
                    {
                        sum += pAccumulators[i] * *pWeights;
                        pWeights += hiddenLayerSize;
                    }

                    hiddenLayer[j] = SCReLU(sum + pInputBiases[j]);
                }

                // Output layer computation
                int output = 0;

                for (int j = 0; j < hiddenLayerSize; j++)
                {
                    output += hiddenLayer[j] * pOutputWeights[j];
                }

                return (output / quantise + OutputBias) * scale / quantiseSquared;
            }
        }
    }


    public static int Evaluate(Board board)
    {
        return board.IsWhiteToMove ? NeuralNetwork.PredictWithAcc() + tempo : -NeuralNetwork.PredictWithAcc() + tempo;
    }

    static readonly float ttSlotSizeMB = 0.000024F;
    public static int hashSizeMB = 1024;
    static int hashSize = Convert.ToInt32(hashSizeMB / ttSlotSizeMB);

    // this tuple is 24 bytes
    static (
        ulong, // hash 8 bytes
        ushort, // moveRaw 4 bytes
        int, // score 4 bytes
        int, // depth 4 bytes
        int // bound BOUND_EXACT=2, BOUND_LOWER=1, BOUND_UPPER=2 4 bytes
    )[] transpositionTable = new (ulong, ushort, int, int, int)[hashSize];

    // Variables for search
    static int infinity = 30000;
    static int mateScore = -20000;

    public static int nodeLimit = 0;

    public static int rfpMargin = 72;
    public static int rfpDepth = 9;
    public static int NullMoveR = 4;
    public static int futilityMargin = 252;
    public static int futilityDepth = 2;
    public static int aspDepth = 2;
    public static int aspDelta = 38;
    public static int lmrMoveCount = 4;
    public static int hardBoundTimeRatio = 3;
    public static int softBoundTimeRatio = 33;
    public static int iirDepth = 7;
    public static int lmrDepth = 1;
    public static float lmrBase = 0.62F;
    public static float lmrMul = 0.4F;
    public static int tempo = 12;
    public static int[] deltas = { 0, 125, 326, 361, 411, 938 };

    public static ulong totalNodes = 0;

    enum ScoreType { upperbound, lowerbound, none };

    int lowerbound = 0;
    int upperbound = 1;
    int exact = 2;

    public static void resetNodes()
    {
        transpositionTable = new (ulong, ushort, int, int, int)[hashSize];
        totalNodes = 0;
    }

    public static void setMargins(int VHashSizeMB, int VrfpMargin, int VrfpDepth, int VfutilityMargin, int VfutilityDepth, int VhardBoundTimeRatio, int VsoftBoundTimeRatio, int VaspDepth, int VaspDelta, int VnullMoveR, int VlmrMoveCount, int ViirDepth, int Vtempo, int VpawnDelta, int VknightDelta, int VbishopDelta, int VrookDelta, int VqueenDelta, int VnodeLimit, int VlmrDepth, int VlmrBase, int VlmrMul)
    {
        hashSizeMB = VHashSizeMB;
        hashSize = Convert.ToInt32(hashSizeMB / ttSlotSizeMB);
        transpositionTable = new (ulong, ushort, int, int, int)[hashSize];

        rfpMargin = VrfpMargin;
        rfpDepth = VrfpDepth;
        futilityMargin = VfutilityMargin;
        futilityDepth = VfutilityDepth;
        hardBoundTimeRatio = VhardBoundTimeRatio;
        softBoundTimeRatio = VsoftBoundTimeRatio;
        aspDepth = VaspDepth;
        aspDelta = VaspDelta;
        iirDepth = ViirDepth;
        tempo = Vtempo;
        deltas[1] = VpawnDelta;
        deltas[2] = VknightDelta;
        deltas[3] = VbishopDelta;
        deltas[4] = VrookDelta;
        deltas[5] = VqueenDelta;
        NullMoveR = VnullMoveR;
        lmrMoveCount = VlmrMoveCount;

        nodeLimit = VnodeLimit;
        lmrDepth = VlmrDepth;
        lmrBase = ((float)VlmrBase) / 100;
        lmrMul = ((float)VlmrMul) / 100;

    }

    public static ulong perftD(Board board, int depth)
    {

        ulong nodes = 0;

        if (depth == 0)
            return 1;

        Move[] legals = board.GetLegalMoves();
        for (int i = 0; i < legals.Length; i++)
        {
            Move move = legals[i];
            board.MakeMove(move);
            nodes += perftD(board, depth - 1);
            board.UndoMove(move);

        }

        return nodes;
    }

    public static void perft(Board board, int maxDepth)
    {
        ulong nodes;
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.Write($"Initiating a perft test for fen ");
        Console.ResetColor();
        Console.WriteLine($"{board.GetFenString()}");

        for (int depth = 1; depth <= maxDepth; depth++)
        {
            var stopwatch = Stopwatch.StartNew();
            nodes = perftD(board, depth);
            stopwatch.Stop();

            double seconds = stopwatch.Elapsed.TotalSeconds;
            double nodesPerSecond = nodes / seconds;

            Console.ForegroundColor = ConsoleColor.Magenta;
            Console.Write($"info ");
            Console.ResetColor();
            Console.WriteLine($"string perft {depth,4} depth {nodes,10} nodes {(ulong)(seconds * 1000),8} time {(ulong)nodesPerSecond,8} nps");
        }
    }

    public Move Think(Board board, Timer timer)
    {

        int selDepth = 0;

        // Killer moves, 1 for each depth
        Move[] killers = new Move[4096];

        // History moves from-to
        int[,] history = new int[64, 64];

        int globalDepth = 1; // To be incremented for each iterative loop
        ulong nodes = 0; // To keep track of searched positions
        Move rootBestMove = Move.NullMove;

        void printInfo(int score, ScoreType scoreType)
        {
            string scoreTypeStr = scoreType == ScoreType.upperbound ? "upperbound " : scoreType == ScoreType.lowerbound ? "lowerbound " : "";

            if (!Console.IsOutputRedirected)
            {
                // Pretty printing with alignment and colored variables
                Console.ForegroundColor = ConsoleColor.Magenta;
                Console.Write("info ");
                Console.ResetColor();

                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.Write($"{globalDepth,4} ");
                Console.ResetColor();
                Console.Write("iteration ");

                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.Write($"{selDepth,4} ");
                Console.ResetColor();
                Console.Write("seldepth ");

                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.Write($"{timer.MillisecondsElapsedThisTurn,8} ");
                Console.ResetColor();
                Console.Write("ms ");

                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.Write($"{nodes,10} ");
                Console.ResetColor();
                Console.Write("nodes ");

                Console.ForegroundColor = ConsoleColor.Green;
                Console.Write($"{(int)(1000 * totalNodes / ((ulong)timer.MillisecondsElapsedThisTurn + 0.001)),8} ");
                Console.ResetColor();
                Console.Write("nps ");

                Console.ForegroundColor = ConsoleColor.Magenta;
                Console.Write($"{100 * nodes / (ulong)hashSize,6} ");
                Console.ResetColor();
                Console.Write("% hashfull ");

                if (score < -30)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                }
                else if (score > 30)
                {
                    Console.ForegroundColor = ConsoleColor.Green;
                }
                else
                {
                    Console.ForegroundColor = ConsoleColor.DarkGray;
                }

                Console.Write($"{((score >= 0 ? "+" : "") + ((float)score / 100).ToString("0.00")),6} ");
                Console.ResetColor();
                Console.Write("score ");


                Console.Write($"{scoreTypeStr}pv ");

                Console.ForegroundColor = ConsoleColor.Blue;
                Console.Write($"{ChessChallenge.Chess.MoveUtility.GetMoveNameUCI(new(rootBestMove.RawValue))}");
                Console.ResetColor();

                Console.WriteLine();
            }

            else
            {
                // Standard printing for non-terminal
                Console.WriteLine($"info depth {globalDepth} seldepth {selDepth} time {timer.MillisecondsElapsedThisTurn} nodes {nodes} nps {(int)(1000 * nodes / ((ulong)timer.MillisecondsElapsedThisTurn + 0.001))} hashfull {1000 * totalNodes / (ulong)hashSize} score cp {score} {scoreTypeStr}pv {ChessChallenge.Chess.MoveUtility.GetMoveNameUCI(new(rootBestMove.RawValue))}");
            }
        }



        int qSearch(int alpha, int beta, int ply)
        {
            // Step 1: Hard bound time management
            if (globalDepth > 1 && timer.MillisecondsElapsedThisTurn >= timer.MillisecondsRemaining / hardBoundTimeRatio) throw null;

            // Step 2: Update selective depth
            if (ply > selDepth) selDepth = ply;

            // Step 3: Check if position is draw or checkmate
            if (board.IsDraw() || board.IsInCheckmate()) return 0;

            // Step 4: TT Cutoffs
            ref var tt = ref transpositionTable[board.ZobristKey & (ulong)(hashSize - 1)];
            var (ttHash, ttMoveRaw, score, ttDepth, ttBound) = tt;
            bool ttHit = ttHash == board.ZobristKey;
            int oldAlpha = alpha;

            if (ttHit)
            {
                if ((ttBound == lowerbound && score >= beta) ||
                    (ttBound == upperbound && score <= alpha) ||
                    (ttBound == exact))
                {
                    return score;
                }
            }

            int standPat = Evaluate(board);
            int bestScore = standPat;

            // Step 5: Standing Pat Pruning
            if (standPat >= beta) return standPat;
            if (alpha < standPat) alpha = standPat;

            Move bestMove = Move.NullMove;

            // Step 6: Move ordering
            // TT + MVV-LVA
            Move[] captures = board.GetLegalMoves(true);
            int[] orderKeys = new int[captures.Length];
            for (int i = 0; i < captures.Length; i++)
            {
                var ormove = captures[i];
                int orderKey = 0;

                if (ttHit && ormove.RawValue == ttMoveRaw)
                {
                    orderKey = 100_000_000;
                }
                else
                {
                    orderKey = 1_000_000 * (int)ormove.CapturePieceType - (int)ormove.MovePieceType;
                }
                orderKeys[i] = orderKey;
            }

            Array.Sort(orderKeys, captures);

            // Main loop
            for (int i = captures.Length - 1; i >= 0; i--)
            {
                Move move = captures[i];

                // Step 7: Delta pruning
                if (standPat + deltas[(int)move.CapturePieceType] < alpha) break;

                nodes++;
                totalNodes++;

                Piece piece = board.GetPiece(move.StartSquare);
                updateAccumulators(piece, move);
                board.MakeMove(move);

                score = -qSearch(-beta, -alpha, ply + 1);

                board.UndoMove(move);
                undoUpdateAccumulators(piece, move);

                if (score > bestScore)
                {
                    // Step 8: Update best move, score and alpha
                    bestMove = move;
                    bestScore = score;
                    if (score > alpha)
                    {
                        alpha = score;
                        // Step 9: Beta pruning
                        if (score >= beta) break;
                    }
                }
            }

            // Step 10: Update TT
            tt = (
                board.ZobristKey,
                alpha > oldAlpha ? bestMove.RawValue : ttMoveRaw,
                Math.Clamp(bestScore, mateScore, -mateScore),
                0,
                bestScore >= beta ? lowerbound : alpha == oldAlpha ? upperbound : exact
            );

            return bestScore;
        }

        // Fail-Soft Negamax Search
        int search(int depth, int ply, int alpha, int beta)
        {

            // Step 1 Hard time limit
            if (depth > 1 && timer.MillisecondsElapsedThisTurn >= timer.MillisecondsRemaining / hardBoundTimeRatio) throw null;

            // Step 2 Update selective depth
            if (ply > selDepth) selDepth = ply;

            bool isRoot = ply == 0;
            bool nonPv = alpha + 1 == beta;

            // Step 3 Mate distancing pruning
            if (!isRoot)
            {
                alpha = Math.Max(alpha, mateScore + ply);
                beta = Math.Min(beta, -mateScore - ply - 1);

                if (alpha >= beta)
                    return alpha;
            }

            // Step 4 Resolving terminal nodes
            if (board.IsDraw() && !isRoot) return 0;
            if (board.IsInCheckmate()) return mateScore + ply;

            // Step 5 TT Pruning
            ref var tt = ref transpositionTable[board.ZobristKey & (ulong)(hashSize - 1)];
            var (ttHash, ttMoveRaw, score, ttDepth, ttBound) = tt;

            bool ttHit = ttHash == board.ZobristKey;
            int oldAlpha = alpha;

            if (nonPv && ttHit)
            {
                if (ttDepth >= depth)
                {
                    if ((ttBound == lowerbound && score >= beta) ||
                        (ttBound == upperbound && score <= alpha) ||
                        (ttBound == exact))
                    {
                        return score;
                    }
                }
            }
            // Step 6 Internal iterative reduction
            else if (!nonPv && depth > iirDepth) depth--;

            // Step 7 Start quiescence search if depth < 1
            if (depth < 1) return qSearch(alpha, beta, ply);

            // Step 8 Static eval needed for RFP and NMP
            int eval = Evaluate(board);

            // Step 9 Index for killers
            int killerIndex = ply & 4095;

            bool nodeIsCheck = board.IsInCheck();
            // Step 10 Reverse futility pruning
            if (nonPv && depth <= rfpDepth && eval - rfpMargin * depth >= beta && !nodeIsCheck) return eval;

            // Step 11 Null move pruning
            if (nonPv && eval >= beta && !nodeIsCheck)
            {
                board.ForceSkipTurn();
                eval = -search(depth - NullMoveR, ply + 1, -beta + 1, -beta);
                board.UndoSkipTurn();

                if (eval >= beta) return eval;
            }

            int bestScore = -infinity;
            int moveCount = 0;
            int quietIndex = 0;

            Move bestMove = Move.NullMove;
            Move[] legals = board.GetLegalMoves();

            (int, int)[] quietsFromTo = new (int, int)[4096];
            Array.Fill(quietsFromTo, (-1, -1));

            // Step 12 Move reordering
            // orderVariable(priority)
            // TT(0),  MVV-LVA ordering(1),  Killer Moves(2)

            int[] orderKeys = new int[legals.Length];
            for (int i = 0; i < legals.Length; i++)
            {
                var ormove = legals[i];
                int orderKey = 0;

                if (ttHit && ormove.RawValue == ttMoveRaw)
                {
                    orderKey = 100_000_000;
                }
                else if (ormove.IsCapture)
                {
                    orderKey = 1_000_000 * (int)ormove.CapturePieceType - (int)ormove.MovePieceType;
                }
                else if (ormove == killers[killerIndex])
                {
                    orderKey = 100_000;
                }
                else
                {
                    orderKey = history[ormove.StartSquare.Index, ormove.TargetSquare.Index];
                }

                orderKeys[i] = orderKey;
            }

            Array.Sort(orderKeys, legals);

            Move move;
            for (int i = legals.Length - 1; i >= 0; i--)
            {
                moveCount++;
                move = legals[i];

                // Budget late moves pruning
                if (legals.Length > 15 && moveCount >= legals.Length - 2 && !move.IsCapture && nonPv) continue;

                bool isQuiet = !move.IsCapture;

                // Step 13 Futility pruning
                if (nonPv && depth <= futilityDepth && isQuiet && (eval + futilityMargin * depth < alpha) && bestScore > mateScore + 100) continue;

                nodes++;
                totalNodes++;

                // Step 14 Late moves reduction
                int reduction = moveCount > lmrMoveCount && depth >= lmrDepth && isQuiet && !nodeIsCheck && nonPv ? (int)(lmrBase + Math.Log(depth) * Math.Log(moveCount) * lmrMul) : 0;

                Piece piece = board.GetPiece(move.StartSquare);
                updateAccumulators(piece, move);

                board.MakeMove(move);

                // Step 15 Check extension
                int moveExtension = board.IsInCheck() ? 1 : 0;

                score = 0;

                // Step 16 Principle variation search
                if (moveCount == 1 && !nonPv)
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
                undoUpdateAccumulators(piece, move);

                // Step 17 Updating best score, best move and alpha
                if (score > bestScore)
                {
                    bestScore = score;
                    bestMove = move;
                    if (isRoot)
                        rootBestMove = move;

                    if (score > alpha)
                    {
                        alpha = score;

                        // Step 18 Beta pruning
                        if (score >= beta)
                        {

                            if (isQuiet)
                            {
                                int bonus = depth * depth;

                                // Step 19 History Malus
                                for (i = 0; i < quietsFromTo.Length; i++)
                                {
                                    var indexes = quietsFromTo[i];
                                    if (indexes.Item1 == -1)
                                        break;
                                    history[indexes.Item1, indexes.Item2] -= bonus + (history[indexes.Item1, indexes.Item2] * bonus / 16384);
                                }

                                // Step 20 History bonus
                                history[move.StartSquare.Index, move.TargetSquare.Index] += bonus - (history[move.StartSquare.Index, move.TargetSquare.Index] * bonus / 16384);

                                // Step 21 Update quiet list for this
                                quietsFromTo[quietIndex] = (move.StartSquare.Index, move.TargetSquare.Index);
                                quietIndex++;

                                // Step 22 Killer moves
                                killers[killerIndex] = move;

                            }
                            break;
                        }
                    }
                }

                // Step 23 Update quiet list
                if (isQuiet)
                {
                    quietsFromTo[quietIndex] = (move.StartSquare.Index, move.TargetSquare.Index);
                    quietIndex++;
                }
            }

            // Step 25 Update TT
            tt = (
                    board.ZobristKey,
                    alpha > oldAlpha ? bestMove.RawValue : ttMoveRaw,
                    Math.Clamp(bestScore, mateScore, -mateScore),
                    depth,
                    bestScore >= beta ? lowerbound /* lowerbound */ : alpha == oldAlpha ? upperbound /* upperbound */ : exact /* Exact */
            );

            return bestScore;

        }


        try
        {
            accumulators = Encode(board.GetFenString());

            nodes = 0;
            int score = 0;
            // Soft bound time limit
            for (; timer.MillisecondsElapsedThisTurn < timer.MillisecondsRemaining / softBoundTimeRatio; ++globalDepth)
            {
                // Soft bound node limit
                if (nodeLimit != 0 && nodes > (ulong)nodeLimit)
                    break;

                int alpha = -infinity;
                int beta = infinity;
                int delta = 0;
                int newScore;

                if (globalDepth > aspDepth)
                {
                    delta = aspDelta;
                    alpha = score - delta;
                    beta = score + delta;
                }

                while (true)
                {
                    newScore = search(globalDepth, 0, alpha, beta);
                    if (newScore <= alpha)
                    {
                        beta = (newScore + beta) / 2;
                        alpha = newScore - delta;

                        printInfo(alpha, ScoreType.upperbound);
                    }
                    else if (newScore >= beta)
                    {
                        beta = newScore + delta;

                        printInfo(beta, ScoreType.lowerbound);
                    }
                    else
                        break;

                    if (delta <= 500 && timer.MillisecondsElapsedThisTurn < timer.MillisecondsRemaining / softBoundTimeRatio)
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