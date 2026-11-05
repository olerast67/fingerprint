// Сравнение отпечатков программой SourceAFIS.
// Запуск: dotnet run -- <dpi> <эталон.png> <кандидат1.png> [кандидат2.png ...]
// Печатает оценку сходства для каждого кандидата. Порог 40 у SourceAFIS
// соответствует одной ложной находке на 10 000 сравнений.
using SourceAFIS;

double dpi = double.Parse(args[0], System.Globalization.CultureInfo.InvariantCulture);
FingerprintTemplate Load(string path) =>
    new FingerprintTemplate(new FingerprintImage(File.ReadAllBytes(path), new FingerprintImageOptions { Dpi = dpi }));

var matcher = new FingerprintMatcher(Load(args[1]));
foreach (var path in args.Skip(2))
    Console.WriteLine($"{Path.GetFileName(path)}\t{matcher.Match(Load(path)):F1}");
