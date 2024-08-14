using System;
using System.Linq;
using System.IO;
using ChessChallenge.API;
using System.Diagnostics;

/* Let it rest
using static ChessChallenge.API.BitboardHelper;
using System.Collections.Generic;
using System.Numerics;
*/
public class MyBot : IChessBot
{

    static readonly int inputLayerSize = 384;
    static readonly int hiddenLayerSize = 32;
    static readonly int scale = 150;
    static readonly int quantise = 255;
    static int[,] FeatureWeights = new int[inputLayerSize, hiddenLayerSize];
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
            FeatureWeights[row, col] = (int)(BitConverter.ToSingle(tmp1, 0) * quantise);

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
        int[] boardArray = new int[384];
        ReadOnlySpan<char> span = fen.AsSpan();
        int spaceIndex = span.IndexOf(' ');
        ReadOnlySpan<char> boardSpan = span.Slice(0, spaceIndex);
        char turn = span[spaceIndex + 1];
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
                    'P' or 'p' => 0,
                    'N' or 'n' => 64,
                    'B' or 'b' => 128,
                    'R' or 'r' => 192,
                    'Q' or 'q' => 256,
                    'K' or 'k' => 320,
                    _ => throw new InvalidOperationException("Invalid piece character")
                };

                int boardPosition = rowIdx * 8 + colIdx;
                bool whiteTurn = turn == 'w';
                int arrayIndex = whiteTurn ? boardPosition : boardPosition ^ 56;

                if (char.IsUpper(character))
                {
                    boardArray[pieceIndex + arrayIndex] = whiteTurn ? 1 : -1;
                }
                else
                {
                    boardArray[pieceIndex + arrayIndex] = whiteTurn ? -1 : 1;
                }
                colIdx++;
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
        private static int SCReLU(int x)
        {
            int clipped = Math.Clamp(x, 0, quantise);
            return clipped * clipped;
        }

        public static int Predict(int[] inputs, int[,] inputWeights, int[] inputBiases, int[] outputWeights, int outputBias)
        {

            // Compute hidden layer activations
            int[] hiddenLayer = new int[hiddenLayerSize];
            for (int j = 0; j < hiddenLayerSize; j++)
            {
                int sum = 0;
                for (int i = 0; i < inputLayerSize; i++)
                {
                    sum += inputs[i] * inputWeights[i, j];
                }
                hiddenLayer[j] = SCReLU(sum + inputBiases[j]);
            }

            // Compute output layer activation
            int output = 0;
            for (int j = 0; j < hiddenLayerSize; j++)
            {
                output += hiddenLayer[j] * outputWeights[j];
            }

            return (output / quantise + outputBias) * scale / (quantise * quantise);
        }

    }

    public static int Evaluate(Board board)
    {
        int[] encoded = Encode(board.GetFenString());
        int prediction = 0;
        prediction = NeuralNetwork.Predict(encoded, FeatureWeights, FeatureBias, OutputWeights, OutputBias);

        return prediction + tempo;
    }

    /* Just going to let this code sleep for now, not sure how to test it yet
    public static int getStaticPieceScore(PieceType pieceType) {
        switch (pieceType) {
            case PieceType.Pawn:
                return 100;
            case PieceType.Knight:
                return 300;
            case PieceType.Bishop:
                return 300;
            case PieceType.Rook:
                return 500;
            case PieceType.Queen:
                return 1000;
            case PieceType.King:
                return 100000;
            default:
                return 0;
        }
    }

    public static class StaticExchangeEvaluation
    {
        private static short[][][] _table;

        public static void Init()
        {
            InitTable();
            PopulateTable();
        }

        public static short Evaluate(Piece attackingPiece, Piece capturedPiece, Piece attacker, Piece defender)
        {
            return (short)(getStaticPieceScore(capturedPiece.PieceType) + _table[(int)attackingPiece.PieceType - 1][(int)attacker.PieceType - 1][(int)defender.PieceType - 1]);
        }

        private static void InitTable()
        {
            _table = new short[6][][];
            for (var attackingPiece = 0; attackingPiece < 6; attackingPiece++)
            {
                _table[attackingPiece] = new short[256][];
                for (var attackerIndex = 0; attackerIndex < 256; attackerIndex++)
                {
                    _table[attackingPiece][attackerIndex] = new short[256];
                }
            }
        }

        private static void PopulateTable()
        {
            var gainList = new List<int>();
            for (var attackingPiece = 0; attackingPiece < 6; attackingPiece++)
            {
                for (ulong attackerIndex = 0; attackerIndex < 256; attackerIndex++)
                {
                    for (ulong defenderIndex = 0; defenderIndex < 256; defenderIndex++)
                    {
                        var attackingPieceSeeIndex = attackingPiece;
                        var attackers = attackerIndex & ~(1ul << attackingPieceSeeIndex);
                        var defenders = defenderIndex;

                        var currentPieceOnField = attackingPiece;
                        var result = 0;

                        gainList.Add(result);

                        if (defenders != 0)
                        {
                            var leastValuableDefenderPiece = GetLeastValuablePiece(defenders);
                            defenders = (ulong)ChessChallenge.Chess.BitBoardUtility.PopLSB(ref defenders);

                            result -= getStaticPieceScore((PieceType)(currentPieceOnField + 1));
                            currentPieceOnField = leastValuableDefenderPiece;

                            gainList.Add(result);

                            while (attackers != 0)
                            {
                                var leastValuableAttackerPiece = GetLeastValuablePiece(attackers);
                                attackers = (ulong)ChessChallenge.Chess.BitBoardUtility.PopLSB(ref attackers);

                                result += getStaticPieceScore((PieceType)(currentPieceOnField + 1));
                                currentPieceOnField = leastValuableAttackerPiece;

                                gainList.Add(result);

                                if (gainList[^1] > gainList[^3])
                                {
                                    result = gainList[^3];
                                    break;
                                }

                                if (defenders != 0)
                                {
                                    leastValuableDefenderPiece = GetLeastValuablePiece(defenders);
                                    defenders = (ulong)ChessChallenge.Chess.BitBoardUtility.PopLSB(ref defenders);

                                    result -= getStaticPieceScore((PieceType)(currentPieceOnField + 1));
                                    currentPieceOnField = leastValuableDefenderPiece;

                                    gainList.Add(result);

                                    if (gainList[^1] < gainList[^3])
                                    {
                                        result = gainList[^3];
                                        break;
                                    }
                                }
                                else
                                {
                                    break;
                                }
                            }
                        }

                        _table[attackingPiece][attackerIndex][defenderIndex] = (short)result;
                        gainList.Clear();
                    }
                }
            }
        }
        private static int GetLeastValuablePiece(ulong data)
        {
            var leastValuableDefenderField = data & 1;
            var leastValuableDefenderPiece = BitOperations.TrailingZeroCount(leastValuableDefenderField);

            return leastValuableDefenderPiece;
        }
    }

    */

    static readonly double ttSlotSizeMB = 0.000024;
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
    public static double lmrBase = 0.75D;
    public static double lmrMul = 0.4D;
    public static int tempo = 14;
    public static int[] deltas = { 200, 500, 600, 1000, 2000 };

    enum ScoreType { upperbound, lowerbound, none };

    public static void setMargins(int VHashSizeMB, int VrfpMargin, int VrfpDepth, int VfutilityMargin, int VfutilityDepth, int VhardBoundTimeRatio, int VsoftBoundTimeRatio, int VaspDepth, int VaspDelta, int VnullMoveR, int VlmrMoveCount, int ViirDepth, int Vtempo, int VpawnDelta, int VknightDelta, int VbishopDelta, int VrookDelta, int VqueenDelta)
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

            // TT + MVV-LVA ordering
            foreach (Move move in board.GetLegalMoves(true).OrderByDescending(move => ttHit && move.RawValue == ttMoveRaw ? 9_000_000_000_000_000_000
                                          : 1_000_000_000_000_000_000 * (long)move.CapturePieceType - (long)move.MovePieceType))
            {
                if (standPat + deltas[(int)move.CapturePieceType - 1] < alpha)
                {
                    break;
                }

                nodes++;
                board.MakeMove(move);
                score = -qSearch(-beta, -alpha, ply + 1);
                board.UndoMove(move);

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
                    Math.Clamp(bestScore, -20000, 20000),
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

            int bestScore = -30000;
            int moveCount = 0;
            int quietIndex = 0;

            // orderVariable(priority)
            // TT(0),  MVV-LVA ordering(1),  Killer Moves(2)

            Move bestMove = Move.NullMove;
            Move[] legals = board.GetLegalMoves();

            (int, int)[] quietsFromTo = new (int, int)[4096];
            Array.Fill(quietsFromTo, (-1, -1));

            foreach (Move move in legals.OrderByDescending(move => ttHit && move.RawValue == ttMoveRaw ? 9_000_000_000_000_000_000
                                          : move.IsCapture ? 1_000_000_000_000_000_000 * (long)move.CapturePieceType - (long)move.MovePieceType
                                          : move == killers[killerIndex] ? 500_000_000_000_000_000
                                          : history[move.StartSquare.Index, move.TargetSquare.Index]))
            {
                if (nonPv && depth <= futilityDepth && !move.IsCapture && (eval + futilityMargin * depth < alpha) && bestScore > mateScore + 100)
                    continue;

                moveCount++;
                nodes++;

                bool isQuiet = !move.IsCapture;

                int reduction = moveCount > lmrCount && depth >= lmrDepth && isQuiet && !nodeIsCheck && nonPv ? (int)(lmrBase + Math.Log(depth) * Math.Log(moveCount) * lmrMul) : 0;

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
                    Math.Clamp(bestScore, -20000, 20000),
                    depth,
                    bestScore >= beta ? 0 /* lowerbound */ : alpha == oldAlpha ? 1 /* upperbound */ : 2 /* Exact */
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