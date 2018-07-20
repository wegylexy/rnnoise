using NAudio.Wave;
using NAudio.Wave.Compression;
using System;
using System.IO;

namespace RnNoiseClr.Demo
{
	class Program
	{
		static void Main(string[] args)
		{
			var fi = new FileInfo(args[0]);
			var fn = fi.FullName;
			using (var wfr = new WaveFileReader(fn))
			{
				var wf0 = wfr.WaveFormat;
				var wf1 = new WaveFormat(48000, 1);
				var size0 = wf0.AverageBytesPerSecond * RNNoiseCLR.FRAME_SIZE / 48000;
				var size1 = sizeof(short) * RNNoiseCLR.FRAME_SIZE;
				using (var acmr = new AcmStream(wf0, wf1))
				using (var wfw = new WaveFileWriter(fn.Substring(0, fn.Length - fi.Extension.Length) + "_out" + fi.Extension, wf0))
				using (var acmw = new AcmStream(wf1, wf0))
				using (var rnn = new RNNoiseCLR())
				{
					var samples = new short[RNNoiseCLR.FRAME_SIZE];
					int read;
					while ((read = wfr.Read(acmr.SourceBuffer, 0, size0)) > 0)
					{
						var converted = acmr.Convert(read, out _);
						for (var i = converted; i < size1; ++i)
							acmr.DestBuffer[i] = 0;
						Buffer.BlockCopy(acmr.DestBuffer, 0, samples, 0, size1);
						rnn.Transform(samples, samples);
						Buffer.BlockCopy(samples, 0, acmw.SourceBuffer, 0, converted);
						wfw.Write(acmw.DestBuffer, 0, acmw.Convert(converted, out _));
						wfw.Flush();
					}
				}
			}
		}
	}
}