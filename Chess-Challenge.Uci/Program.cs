﻿namespace Chess_Challenge.Cli
{
    internal class Program
    {

        static void Main(string[] args)
        {
            if (!Console.IsOutputRedirected)
            {
                Console.ForegroundColor = ConsoleColor.Magenta;
                Console.WriteLine("+---------------------------------------------------+");
                Console.Write("| Customised ");
                Console.ResetColor();
                Console.WriteLine("pretty-print user interface for Nectar |");
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.Write("| Version    ");
                Console.ResetColor();
                Console.WriteLine("0.2.2                                  |");
                Console.WriteLine("+---------------------------------------------------+");
            }
            var uci = new Uci();
            uci.Run();
        }
    }
}