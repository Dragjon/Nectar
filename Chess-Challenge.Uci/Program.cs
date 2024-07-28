using ChessChallenge.Application;

namespace Chess_Challenge.Cli
{
    internal class Program
    {

        static void Main(string[] args)
        {
            var uci = new Uci();
            uci.Run();
        }
    }
}