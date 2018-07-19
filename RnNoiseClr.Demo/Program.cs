using NAudio.Wave;
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
				var wf = wfr.WaveFormat;
				using (var mfrr = new MediaFoundationResampler(wfr, 48000))
				using (var wfw = new WaveFileWriter(fn.Substring(0, fn.Length - fi.Extension.Length) + "_out" + fi.Extension, mfrr.WaveFormat))
				using (var rnn = new RNNoiseCLR())
				{
					var size = mfrr.WaveFormat.AverageBytesPerSecond * RNNoiseCLR.FRAME_SIZE / 48000;
					var buffer = new byte[size];
					var samples = new short[RNNoiseCLR.FRAME_SIZE];
					var denoised = new short[samples.Length];
					int read;
					while ((read = mfrr.Read(buffer, 0, size)) > 0)
					{
						for (var i = read; i < size; ++i)
							buffer[i] = 0;
						Buffer.BlockCopy(buffer, 0, samples, 0, size);
						rnn.Transform(samples, denoised);
						Buffer.BlockCopy(denoised, 0, buffer, 0, read);
						//using (var rsws = new RawSourceWaveStream(buffer, 0, read, mfrr.WaveFormat))
						//using (var mfrw = new MediaFoundationResampler(rsws, wf))
						//{
						//	while ((read = mfrw.Read(buffer, 0, size)) > 0)
						//		wfw.Write(buffer, 0, read);
						//	wfw.Flush();
						//}
						wfw.Write(buffer, 0, read);
						wfw.Flush();
					}
				}
			}
		}
	}
}
