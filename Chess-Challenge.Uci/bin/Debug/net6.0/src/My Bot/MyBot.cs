using System;
using System.Collections;
using System.IO;
using ChessChallenge.API;
public class MyBot : IChessBot
{

    static readonly int inputLayerSize = 384;
    static readonly int hiddenLayerSize = 32;

    static float[,] FeatureWeights = new float[inputLayerSize, hiddenLayerSize];
    static float[] FeatureBias = new float[hiddenLayerSize];
    static float[] OutputWeights = new float[hiddenLayerSize];
    static float OutputBias;

    public static int SetWeights()
    {
        byte[] iweightsbytes = File.ReadAllBytes("./iweights.bin");
        byte[] ibiasesbytes = File.ReadAllBytes("./ibiases.bin");
        byte[] oweightsbytes = File.ReadAllBytes("./oweights.bin");
        byte[] obiasesbytes = File.ReadAllBytes("./obiases.bin");

        int row = 0;
        int col = 0;

        // Input weights
        for (int i = 0; i < inputLayerSize * hiddenLayerSize * 4; i += 4)
        {
            byte[] tmp1 = new byte[] { iweightsbytes[i], iweightsbytes[i + 1], iweightsbytes[i + 2], iweightsbytes[i + 3] };
            FeatureWeights[row, col] = BitConverter.ToSingle(tmp1, 0);

            col++;
            if (col + 1 == hiddenLayerSize)
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
            FeatureBias[col] = BitConverter.ToSingle(tmp2, 0);
            col++;
        }

        // Output weights
        col = 0;
        for (int i = 0; i < hiddenLayerSize * 4; i += 4)
        {
            byte[] tmp3 = new byte[] { oweightsbytes[i], oweightsbytes[i + 1], oweightsbytes[i + 2], oweightsbytes[i + 3] };
            OutputWeights[col] = BitConverter.ToSingle(tmp3, 0);
            col++;
        }

        // Output bias
        byte[] tmp4 = new byte[] { obiasesbytes[0], obiasesbytes[1], obiasesbytes[2], obiasesbytes[3] };
        OutputBias = BitConverter.ToSingle(tmp4, 0);

        return 0;
    }

    public Move Think(Board board, Timer timer)
    {
        SetWeights();
        Console.WriteLine($"Feature Weights Length: {FeatureWeights.Length}");
        Console.WriteLine($"Feature Bias Length: {FeatureBias.Length}");
        Console.WriteLine($"Output Weights Length: {OutputWeights.Length}");
        Move[] moves = board.GetLegalMoves();
        return moves[0];
    }
}