using System.Text;
using ChessChallenge.API;
using ChessChallenge.Chess;
using Board = ChessChallenge.API.Board;
using Move = ChessChallenge.API.Move;
using Timer = ChessChallenge.API.Timer;

namespace Chess_Challenge.Cli
{
    internal class Uci
    {
        private const string StartposFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

        private IChessBot _bot;
        private Board _board;
        private int _hashSizeMB;
        private int _rfpMargin;
        private int _rfpDepth;
        private int _futilityMargin;
        private int _futilityDepth;
        private int _hardBoundTimeRatio;
        private int _softBoundTimeRatio;
        private int _aspDepth;
        private int _aspDelta;
        private int _nullMoveR;
        private int _tempo;
        private int _lmrMoveCount;
        private int _iirDepth;
        private int _lmrDepth;
        private int _pawnDelta;
        private int _knightDelta;
        private int _bishopDelta;
        private int _rookDelta;
        private int _queenDelta;
        private int _nodeLimit;
        private int _lmrBase;
        private int _lmrMul;
        private int _lmpDepth;
        private int _lmpDMul;
        private int _scale;

        public Uci()
        {
            Reset();
        }

        private void Reset()
        {
            _bot = new MyBot();
            _board = Board.CreateBoardFromFEN(StartposFen);
            MyBot.resetNodes();
            _hashSizeMB = MyBot.hashSizeMB;
            _rfpMargin = MyBot.rfpMargin;
            _rfpDepth = MyBot.rfpDepth;
            _futilityMargin = MyBot.futilityMargin;
            _futilityDepth = MyBot.futilityDepth;
            _hardBoundTimeRatio = MyBot.hardBoundTimeRatio;
            _softBoundTimeRatio = MyBot.softBoundTimeRatio;
            _aspDepth = MyBot.aspDepth;
            _aspDelta = MyBot.aspDelta;
            _tempo = MyBot.tempo;
            _lmrMoveCount = MyBot.lmrMoveCount;
            _iirDepth = MyBot.iirDepth;
            _pawnDelta = MyBot.deltas[1];
            _knightDelta = MyBot.deltas[2];
            _bishopDelta = MyBot.deltas[3];
            _rookDelta = MyBot.deltas[4];
            _queenDelta = MyBot.deltas[5];
            _nullMoveR = MyBot.NullMoveR;
            _nodeLimit = MyBot.nodeLimit;
            _lmrDepth = MyBot.lmrDepth;
            _lmrBase = (int)(MyBot.lmrBase * 100);
            _lmrMul = (int)(MyBot.lmrMul * 100);
            _lmpDepth = MyBot.lmpDepth;
            _lmpDMul = MyBot.lmpDMul;
            _scale = MyBot.scale;
        }

        private void HandleUci()
        {
            if (!Console.IsOutputRedirected)
            {
                Console.ForegroundColor = ConsoleColor.Magenta;
                Console.Write("id ");
                Console.ResetColor();
                Console.WriteLine("name Nectar");

                Console.ForegroundColor = ConsoleColor.Magenta;
                Console.Write("id ");
                Console.ResetColor();
                Console.WriteLine("author Dragjon");

                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.Write("option ");
                Console.ResetColor();
                Console.WriteLine($"name Hash type spin      {_hashSizeMB,5} default {1,5} min {1024,8} max");

                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.Write("option ");
                Console.ResetColor();
                Console.WriteLine($"name Threads type spin   {1,5} default {1,5} min {1,8} max");

                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.Write("option ");
                Console.ResetColor();
                Console.WriteLine($"name nodeLimit type spin {_nodeLimit,5} default {0,5} min {1000000,8} max");
            }
            else
            {
                Console.WriteLine("id name Nectar");
                Console.WriteLine("id author Dragjon");
                Console.WriteLine($"option name Hash type spin default {_hashSizeMB} min 1 max 1024");
                Console.WriteLine($"option name Threads type spin default 1 min 1 max 1");
                Console.WriteLine($"option name nodeLimit type spin default {_nodeLimit} min 0 max 1000000");
                /*
                Console.WriteLine($"option name rfpMargin type spin default {_rfpMargin} min 0 max 200");
                Console.WriteLine($"option name rfpDepth type spin default {_rfpDepth} min 0 max 15");
                Console.WriteLine($"option name futilityMargin type spin default {_futilityMargin} min 0 max 400");
                Console.WriteLine($"option name futilityDepth type spin default {_futilityDepth} min 0 max 10");
                Console.WriteLine($"option name hardBoundTimeRatio type spin default {_hardBoundTimeRatio} min 1 max 100");
                Console.WriteLine($"option name softBoundTimeRatio type spin default {_softBoundTimeRatio} min 1 max 300");
                Console.WriteLine($"option name aspDepth type spin default {_aspDepth} min 0 max 10");
                Console.WriteLine($"option name aspDelta type spin default {_aspDelta} min 0 max 100");
                Console.WriteLine($"option name tempo type spin default {_tempo} min 0 max 40");
                Console.WriteLine($"option name lmrMoveCount type spin default {_lmrMoveCount} min 0 max 10");
                Console.WriteLine($"option name iirDepth type spin default {_iirDepth} min 0 max 10");
                Console.WriteLine($"option name pawnDelta type spin default {_pawnDelta} min 0 max 400");
                Console.WriteLine($"option name knightDelta type spin default {_knightDelta} min 0 max 900");
                Console.WriteLine($"option name bishopDelta type spin default {_bishopDelta} min 0 max 1000");
                Console.WriteLine($"option name rookDelta type spin default {_rookDelta} min 0 max 2000");
                Console.WriteLine($"option name queenDelta type spin default {_queenDelta} min 0 max 5000");
                Console.WriteLine($"option name nullMoveR type spin default {_nullMoveR} min 0 max 10");
                Console.WriteLine($"option name lmrDepth type spin default {_lmrDepth} min 0 max 8");
                Console.WriteLine($"option name lmrBase type spin default {_lmrBase} min 0 max 500");
                Console.WriteLine($"option name lmrMul type spin default {_lmrMul} min 0 max 300");
                Console.WriteLine($"option name lmpDepth type spin default {_lmpDepth} min 0 max 8");
                Console.WriteLine($"option name lmpDMul type spin default {_lmpDMul} min 0 max 30");
                Console.WriteLine($"option name scale type spin default {_scale} min 10 max 1000");
                */


            }

            Console.WriteLine("uciok");
        }

        private void HandleSetOption(IReadOnlyList<string> words)
        {
            if (words.Count < 5) return;

            string optionName = words[1];
            if (optionName == "name" && words[2] == "Hash" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var hashValue))
                {
                    _hashSizeMB = hashValue;
                    Console.WriteLine($"info string Hash set to {_hashSizeMB}");
                }
            }
            else if (optionName == "name" && words[2] == "rfpMargin" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var rfpValue))
                {
                    _rfpMargin = rfpValue;
                    Console.WriteLine($"info string rfpMargin set to {_rfpMargin}");
                }
            }
            else if (optionName == "name" && words[2] == "rfpDepth" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var rfpValue))
                {
                    _rfpDepth = rfpValue;
                    Console.WriteLine($"info string rfpDepth set to {_rfpDepth}");
                }
            }
            else if (optionName == "name" && words[2] == "futilityMargin" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var fpValue))
                {
                    _futilityMargin = fpValue;
                    Console.WriteLine($"info string futilityMargin set to {_futilityMargin}");
                }
            }
            else if (optionName == "name" && words[2] == "futilityDepth" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var fpValue))
                {
                    _futilityDepth = fpValue;
                    Console.WriteLine($"info string futilityDepth set to {_futilityDepth}");
                }
            }
            else if (optionName == "name" && words[2] == "hardBoundTimeRatio" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var hbtr))
                {
                    _hardBoundTimeRatio = hbtr;
                    Console.WriteLine($"info string hardBoundTimeRatio set to {_hardBoundTimeRatio}");
                }
            }
            else if (optionName == "name" && words[2] == "softBoundTimeRatio" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var sbtr))
                {
                    _softBoundTimeRatio = sbtr;
                    Console.WriteLine($"info string softBoundTimeRatio set to {_softBoundTimeRatio}");
                }
            }

            else if (optionName == "name" && words[2] == "aspDepth" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var aspdh))
                {
                    _aspDepth = aspdh;
                    Console.WriteLine($"info string aspDepth set to {_aspDepth}");
                }
            }

            else if (optionName == "name" && words[2] == "aspDelta" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var aspda))
                {
                    _aspDelta = aspda;
                    Console.WriteLine($"info string aspDelta set to {_aspDelta}");
                }
            }

            else if (optionName == "name" && words[2] == "tempo" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var tempo))
                {
                    _tempo = tempo;
                    Console.WriteLine($"info string aspDelta set to {_tempo}");
                }
            }

            else if (optionName == "name" && words[2] == "lmrMoveCount" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var lmrmc))
                {
                    _lmrMoveCount = lmrmc;
                    Console.WriteLine($"info string lmrMoveCount set to {_lmrMoveCount}");
                }
            }

            else if (optionName == "name" && words[2] == "iirDepth" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var iird))
                {
                    _iirDepth = iird;
                    Console.WriteLine($"info string iirDepth set to {_iirDepth}");
                }
            }

            else if (optionName == "name" && words[2] == "pawnDelta" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var pd))
                {
                    _pawnDelta = pd;
                    Console.WriteLine($"info string pawnDelta set to {_pawnDelta}");
                }
            }

            else if (optionName == "name" && words[2] == "knightDelta" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var nd))
                {
                    _knightDelta = nd;
                    Console.WriteLine($"info string knightDelta set to {_knightDelta}");
                }
            }

            else if (optionName == "name" && words[2] == "bishopDelta" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var bd))
                {
                    _bishopDelta = bd;
                    Console.WriteLine($"info string bishopDelta set to {_bishopDelta}");
                }
            }

            else if (optionName == "name" && words[2] == "rookDelta" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var rd))
                {
                    _rookDelta = rd;
                    Console.WriteLine($"info string rookDelta set to {_rookDelta}");
                }
            }

            else if (optionName == "name" && words[2] == "queenDelta" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var qd))
                {
                    _queenDelta = qd;
                    Console.WriteLine($"info string queenDelta set to {_queenDelta}");
                }
            }

            else if (optionName == "name" && words[2] == "nullMoveR" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var nmr))
                {
                    _nullMoveR = nmr;
                    Console.WriteLine($"info string nullMoveR set to {_nullMoveR}");
                }
            }
            else if (optionName == "name" && words[2] == "Threads" && words[3] == "value")
            {
                Console.WriteLine($"info string Threads set to 1");
            }
            else if (optionName == "name" && words[2] == "nodeLimit" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var nlm))
                {
                    _nodeLimit = nlm;
                    Console.WriteLine($"info string nodeLimit set to {_nodeLimit}");
                }
            }
            else if (optionName == "name" && words[2] == "lmrDepth" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var lmrd))
                {
                    _lmrDepth = lmrd;
                    Console.WriteLine($"info string lmrDepth set to {_lmrDepth}");
                }
            }

            else if (optionName == "name" && words[2] == "lmrBase" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var lmrb))
                {
                    _lmrBase = lmrb;
                    Console.WriteLine($"info string lmrBase set to {_lmrBase}");
                }
            }

            else if (optionName == "name" && words[2] == "lmrMul" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var lmrm))
                {
                    _lmrBase = lmrm;
                    Console.WriteLine($"info string lmrMul set to {_lmrMul}");
                }
            }

            else if (optionName == "name" && words[2] == "lmpDepth" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var lmpd))
                {
                    _lmpDepth = lmpd;
                    Console.WriteLine($"info string lmpDepth set to {_lmpDepth}");
                }
            }

            else if (optionName == "name" && words[2] == "lmpDMul" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var lmpdm))
                {
                    _lmpDMul = lmpdm;
                    Console.WriteLine($"info string lmpDMul set to {_lmpDMul}");
                }
            }

            else if (optionName == "name" && words[2] == "scale" && words[3] == "value")
            {
                if (int.TryParse(words[4], out var sc))
                {
                    _scale = sc;
                    Console.WriteLine($"info string scale set to {_scale}");
                }
            }

            MyBot.setMargins(_hashSizeMB, _rfpMargin, _rfpDepth, _futilityMargin, _futilityDepth, _hardBoundTimeRatio, _softBoundTimeRatio, _aspDepth, _aspDelta, _nullMoveR, _lmrMoveCount, _iirDepth, _tempo, _pawnDelta, _knightDelta, _bishopDelta, _rookDelta, _queenDelta, _nodeLimit, _lmrDepth, _lmrBase, _lmrMul, _lmpDepth, _lmpDMul, _scale);

        }

        private void HandlePerft(IReadOnlyList<string> words)
        {
            if (words.Count < 3) return;

            if (words[1] == "depth")
            {
                if (int.TryParse(words[2], out var perftDepth))
                {
                    MyBot.perft(_board, perftDepth);
                }
            }

        }

        private void HandlePosition(IReadOnlyList<string> words)
        {
            var writingFen = false;
            var writingMoves = false;
            var fenBuilder = new StringBuilder();

            for (var wordIndex = 0; wordIndex < words.Count; wordIndex++)
            {
                var word = words[wordIndex];

                if (word == "startpos")
                {
                    _board = Board.CreateBoardFromFEN(StartposFen);
                }

                if (word == "fen")
                {
                    writingFen = true;
                    continue;
                }

                if (word == "moves")
                {
                    if (writingFen)
                    {
                        fenBuilder.Length--;
                        var fen = fenBuilder.ToString();
                        _board = Board.CreateBoardFromFEN(fen);
                    }
                    writingFen = false;
                    writingMoves = true;
                    continue;
                }

                if (writingFen)
                {
                    fenBuilder.Append(word);
                    fenBuilder.Append(' ');
                }

                if (writingMoves)
                {
                    var move = new Move(word, _board);
                    _board.MakeMove(move);
                }
            }

            if (writingFen)
            {
                fenBuilder.Length--;
                var fen = fenBuilder.ToString();
                _board = Board.CreateBoardFromFEN(fen);
            }
        }

        private static string GetMoveName(Move move)
        {
            if (move.IsNull)
            {
                return "Null";
            }

            string startSquareName = BoardHelper.SquareNameFromIndex(move.StartSquare.Index);
            string endSquareName = BoardHelper.SquareNameFromIndex(move.TargetSquare.Index);
            string moveName = startSquareName + endSquareName;
            if (move.IsPromotion)
            {
                switch (move.PromotionPieceType)
                {
                    case PieceType.Rook:
                        moveName += "r";
                        break;
                    case PieceType.Knight:
                        moveName += "n";
                        break;
                    case PieceType.Bishop:
                        moveName += "b";
                        break;
                    case PieceType.Queen:
                        moveName += "q";
                        break;
                }
            }
            return moveName;
        }

        private void HandleGo(IReadOnlyList<string> words)
        {
            var ms = 30000;

            for (var wordIndex = 0; wordIndex < words.Count; wordIndex++)
            {
                var word = words[wordIndex];
                if (words.Count > wordIndex + 1)
                {
                    var nextWord = words[wordIndex + 1];
                    if (word == "wtime" && _board.IsWhiteToMove)
                    {
                        if (int.TryParse(nextWord, out var wtime))
                        {
                            ms = Math.Abs(wtime);
                        }
                    }
                    if (word == "btime" && !_board.IsWhiteToMove)
                    {
                        if (int.TryParse(nextWord, out var btime))
                        {
                            ms = Math.Abs(btime);
                        }
                    }
                }

                if (word == "infinite")
                {
                    ms = int.MaxValue;
                }
            }

            var timer = new Timer(ms);
            var move = _bot.Think(_board, timer);
            var moveStr = GetMoveName(move);
            Console.WriteLine($"bestmove {moveStr}");
        }

        private void HandleLine(string line)
        {
            var words = line.Split(' ');
            if (words.Length == 0)
            {
                return;
            }

            var firstWord = words[0];
            switch (firstWord)
            {
                case "uci":
                    HandleUci();
                    return;
                case "ucinewgame":
                    Reset();
                    return;
                case "position":
                    HandlePosition(words);
                    return;
                case "isready":
                    Console.WriteLine("readyok");
                    return;
                case "go":
                    HandleGo(words);
                    return;
                case "setoption":
                    HandleSetOption(words);
                    return;
                case "seval":
                    Console.WriteLine(MyBot.Evaluate(_board));
                    return;
                case "perft":
                    HandlePerft(words);
                    return;
            }
        }

        public void Run()
        {
            while (true)
            {
                var line = Console.ReadLine();
                if (line == "quit" || line == "exit")
                {
                    return;
                }

                HandleLine(line);
            }
        }
    }
}
