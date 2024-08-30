using System;
using System.Linq;
using System.IO;
using ChessChallenge.API;
using System.Diagnostics;
public class MyBot : IChessBot
{

    static readonly int inputLayerSize = 768;
    static readonly int hiddenLayerSize = 22;
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

        foreach (var character in rows)
        {
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
                    int loopEnd = inputLayerSize - (inputLayerSize % 4);

                    // SIMD-like unrolling (4x at a time)
                    for (; i < loopEnd; i += 4)
                    {
                        sum += pInputs[i] * pWeights[0] +
                               pInputs[i + 1] * pWeights[hiddenLayerSize] +
                               pInputs[i + 2] * pWeights[2 * hiddenLayerSize] +
                               pInputs[i + 3] * pWeights[3 * hiddenLayerSize];

                        pWeights += 4 * hiddenLayerSize;
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
                // Process the hidden layer with loop unrolling and SIMD.
                for (int j = 0; j < hiddenLayerSize; j++)
                {
                    int sum = 0;
                    int* pWeights = pInputWeights + j;

                    int i = 0;
                    int loopEnd = inputLayerSize - (inputLayerSize % 4);

                    // SIMD-like unrolling (4x at a time)
                    for (; i < loopEnd; i += 4)
                    {
                        sum += pAccumulators[i] * pWeights[0] +
                               pAccumulators[i + 1] * pWeights[hiddenLayerSize] +
                               pAccumulators[i + 2] * pWeights[2 * hiddenLayerSize] +
                               pAccumulators[i + 3] * pWeights[3 * hiddenLayerSize];

                        pWeights += 4 * hiddenLayerSize;
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
        if (board.IsWhiteToMove)
        {
            return NeuralNetwork.PredictWithAcc() + tempo;
        }
        else
        {
            return -NeuralNetwork.PredictWithAcc() + tempo;
        }
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

    public static int nodeLimit = -1;

    public static int rfpMargin = 85;
    public static int rfpDepth = 9;
    public static int NullMoveR = 4;
    public static int futilityMargin = 250;
    public static int futilityDepth = 4;
    public static int aspDepth = 3;
    public static int aspDelta = 35;
    public static int lmrMoveCount = 3;
    public static int hardBoundTimeRatio = 1;
    public static int softBoundTimeRatio = 20;
    public static int iirDepth = 3;
    public static int lmrCount = 5;
    public static int lmrDepth = 2;
    public static float lmrBase = 0.75F;
    public static float lmrMul = 0.4F;
    public static int tempo = 14;
    public static int[] deltas = { 90, 340, 350, 410, 930 };

    enum ScoreType { upperbound, lowerbound, none };

    public static void setMargins(int VHashSizeMB, int VrfpMargin, int VrfpDepth, int VfutilityMargin, int VfutilityDepth, int VhardBoundTimeRatio, int VsoftBoundTimeRatio, int VaspDepth, int VaspDelta, int VnullMoveR, int VlmrMoveCount, int ViirDepth, int Vtempo, int VpawnDelta, int VknightDelta, int VbishopDelta, int VrookDelta, int VqueenDelta, int VnodeLimit)
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
        deltas[0] = VpawnDelta;
        deltas[1] = VknightDelta;
        deltas[2] = VbishopDelta;
        deltas[3] = VrookDelta;
        deltas[4] = VqueenDelta;
        NullMoveR = VnullMoveR;
        lmrMoveCount = VlmrMoveCount;

        nodeLimit = VnodeLimit;

    }

    public static ulong perftD(Board board, int depth)
    {

        ulong nodes = 0;

        if (depth == 0)
            return 1;

        Move[] legals = board.GetLegalMoves();
        foreach (var move in legals)
        {
            board.MakeMove(move);
            nodes += perftD(board, depth - 1);
            board.UndoMove(move);

        }

        return nodes;
    }

    public static void perft(Board board, int maxDepth)
    {
        ulong nodes;
        Console.WriteLine($"Initiating a perft test for fen {board.GetFenString()}");

        for (int depth = 1; depth <= maxDepth; depth++)
        {
            var stopwatch = Stopwatch.StartNew();
            nodes = perftD(board, depth);
            stopwatch.Stop();

            double seconds = stopwatch.Elapsed.TotalSeconds;
            double nodesPerSecond = nodes / seconds;

            Console.WriteLine($"info string perft depth {depth} nodes {nodes} time {(ulong)(seconds * 1000)} nps {(ulong)nodesPerSecond}");
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
        ulong nodes = 0; // To keep track of searched positions in 1 iterative loop
        Move rootBestMove = Move.NullMove;

        void printInfo(int score, ScoreType scoreType)
        {
            string scoreTypeStr = scoreType == ScoreType.upperbound ? "upperbound " : scoreType == ScoreType.lowerbound ? "lowerbound " : "";

            Console.WriteLine($"info depth {globalDepth} seldepth {selDepth} time {timer.MillisecondsElapsedThisTurn} nodes {nodes} nps {(int)(1000 * nodes / ((ulong)timer.MillisecondsElapsedThisTurn + 0.001))} hashfull {1000 * nodes / (ulong)hashSize} score cp {score} {scoreTypeStr}pv {ChessChallenge.Chess.MoveUtility.GetMoveNameUCI(new(rootBestMove.RawValue))}");
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
                if (ttBound == 0)
                    if (score >= beta)
                        return score;
                    else if (ttBound == 1)
                        if (score <= alpha)
                            return score;
                        else
                            return score;
            }

            // Standing Pat Pruning
            if (standPat >= beta)
                return standPat;

            if (alpha < standPat)
                alpha = standPat;

            Move bestMove = Move.NullMove;

            Move[] captures = board.GetLegalMoves(true);
            captures = captures.OrderByDescending(move => ttHit && move.RawValue == ttMoveRaw ? 9_000_000_000_000_000_000
                                          : 1_000_000_000_000_000_000 * (long)move.CapturePieceType - (long)move.MovePieceType).ToArray();
            Move move;
            // TT + MVV-LVA ordering
            for (int i = 0; i < captures.Length; i++)
            {
                move = captures[i];
                if (standPat + deltas[(int)move.CapturePieceType - 1] < alpha)
                {
                    break;
                }

                nodes++;

                Piece piece = board.GetPiece(move.StartSquare);
                updateAccumulators(piece, move);

                board.MakeMove(move);
                score = -qSearch(-beta, -alpha, ply + 1);
                board.UndoMove(move);
                undoUpdateAccumulators(piece, move);

                if (score > bestScore)
                {
                    bestMove = move;
                    bestScore = score;
                    if (score > alpha)
                    {
                        alpha = score;
                        // A/B pruning
                        if (score >= beta)
                            break;
                    }
                }

            }

            tt = (
                    board.ZobristKey,
                    alpha > oldAlpha ? bestMove.RawValue : ttMoveRaw,
                    Math.Clamp(bestScore, mateScore, -mateScore),
                    0,
                    bestScore >= beta ? 0 /* lowerbound */ : alpha == oldAlpha ? 1 /* upperbound */ : 2 /* Exact */
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
                if (ttDepth >= depth)
                {
                    if (ttBound == 0)
                        if (score >= beta)
                            return score;
                        else if (ttBound == 1)
                            if (score <= alpha)
                                return score;
                            else
                                return score;
                }
            }
            else if (!nonPv && depth > iirDepth)
                // Internal iterative reduction
                depth--;

            // Start Quiescence Search
            if (depth < 1)
                return qSearch(alpha, beta, ply);

            // Static eval needed for RFP and NMP
            int eval = Evaluate(board);

            // Index for killers
            int killerIndex = ply & 4095;

            bool nodeIsCheck = board.IsInCheck();
            // Reverse futility pruning
            if (nonPv && depth <= rfpDepth && eval - rfpMargin * depth >= beta && !nodeIsCheck) return eval;

            // Null move pruning
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

            // orderVariable(priority)
            // TT(0),  MVV-LVA ordering(1),  Killer Moves(2)

            Move bestMove = Move.NullMove;
            Move[] legals = board.GetLegalMoves();

            (int, int)[] quietsFromTo = new (int, int)[4096];
            Array.Fill(quietsFromTo, (-1, -1));

            legals = legals.OrderByDescending(move => ttHit && move.RawValue == ttMoveRaw ? 9_000_000_000_000_000_000
                                          : move.IsCapture ? 1_000_000_000_000_000_000 * (long)move.CapturePieceType - (long)move.MovePieceType
                                          : move == killers[killerIndex] ? 500_000_000_000_000_000
                                          : history[move.StartSquare.Index, move.TargetSquare.Index]).ToArray();

            Move move;
            for (int i = 0; i < legals.Length; i++)
            {
                move = legals[i];
                bool isQuiet = !move.IsCapture;

                if (nonPv && depth <= futilityDepth && isQuiet && (eval + futilityMargin * depth < alpha) && bestScore > mateScore + 100)
                    continue;

                if (moveCount > 3 + depth * depth && isQuiet && nonPv)
                    continue;

                moveCount++;
                nodes++;


                int reduction = moveCount > lmrCount && depth >= lmrDepth && isQuiet && !nodeIsCheck && nonPv ? (int)(lmrBase + Math.Log(depth) * Math.Log(moveCount) * lmrMul) : 0;
               
                Piece piece = board.GetPiece(move.StartSquare);
                updateAccumulators(piece, move);

                board.MakeMove(move);

                // Check extension
                int moveExtension = board.IsInCheck() ? 1 : 0;

                score = 0;


                // Principle variation search
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

                // Updating stuff
                if (score > bestScore)
                {
                    bestScore = score;
                    bestMove = move;
                    if (isRoot)
                        rootBestMove = move;

                    if (score > alpha)
                    {
                        alpha = score;

                        // A/B pruning
                        if (score >= beta)
                        {

                            if (isQuiet)
                            {
                                int bonus = depth * depth;

                                // History Malus
                                foreach (var indexes in quietsFromTo)
                                {
                                    if (indexes.Item1 == -1)
                                        break;
                                    history[indexes.Item1, indexes.Item2] -= bonus + (history[indexes.Item1, indexes.Item2] * bonus / 16384);
                                }

                                // History bonus
                                history[move.StartSquare.Index, move.TargetSquare.Index] += bonus - (history[move.StartSquare.Index, move.TargetSquare.Index] * bonus / 16384);

                                // Update quiet list for this
                                quietsFromTo[quietIndex] = (move.StartSquare.Index, move.TargetSquare.Index);
                                quietIndex++;

                                // Killer moves
                                killers[killerIndex] = move;

                            }
                            break;
                        }
                    }
                }

                // Update quiet list
                if (isQuiet)
                {
                    quietsFromTo[quietIndex] = (move.StartSquare.Index, move.TargetSquare.Index);
                    quietIndex++;
                }
            }

            tt = (
                    board.ZobristKey,
                    alpha > oldAlpha ? bestMove.RawValue : ttMoveRaw,
                    Math.Clamp(bestScore, mateScore, -mateScore),
                    depth,
                    bestScore >= beta ? 0 /* lowerbound */ : alpha == oldAlpha ? 1 /* upperbound */ : 2 /* Exact */
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
                if (globalDepth > aspDepth)
                {
                    delta = aspDelta;
                    alpha = score - delta;
                    beta = score + delta;
                }
                int newScore;
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