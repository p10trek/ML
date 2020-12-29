using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Drawing;
using System.Windows.Interop;
using System.Diagnostics;
using System.Windows.Threading;

namespace ML
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
     partial class MainWindow : Window
     {
        public delegate void UpdateTextCallback(string message);
        public delegate void UpdateImageCallback(Bitmap bitmap);
        private int line;
        private string pixelFile = @"train-images.idx3-ubyte";
        private string labelFile = @"train-labels.idx1-ubyte";
        private string TestpixelFile = @"t10k-images.idx3-ubyte";
        private string TestlabelFile = @"t10k-labels.idx1-ubyte";
        private double LernRate = 5;
        private int pixelCount = 784;
        private double[,] ImagesPixels = new double[60000,784];
        private double[,] weighth = new double[784,160];
        private double[] biash = new double[160];
        private double[] OutH1 = new double[160];
        private int[] hiddenLayer1 = new int[160];
        private double[,] weighto = new double[160,10];
        private double[] biaso = new double[10];
        private double[] outputs = new double[10];
        private double ErrorTotalOutputs;
        private double[] labels = new double[60000];

        double[] EtotOuts = new double[10];
        double[] OutNets = new double[10];
        double[,] EtotNets = new double[10,160];
        double[] ETotalOut = new double[160];
        double[] OutNetH = new double[160];
        public DigitImage[] trainImages = null;
        public DigitImage[] testImages = null;
        public MainWindow()
        {
            InitializeComponent();
        }

        public void Button_Click(object sender, RoutedEventArgs e)
        {
            this.trainImages = LoadData(pixelFile, labelFile,60000);
            this.testImages = LoadData(TestpixelFile, TestlabelFile, 10000);
        }
        public static DigitImage[] LoadData(string pixelFile, string labelFile,int numImages)
        {

            DigitImage[] result = new DigitImage[numImages];
            byte[][] pixels = new byte[28][];
            for (int i = 0; i < pixels.Length; ++i)
                pixels[i] = new byte[28];
            FileStream ifsPixels = new FileStream(pixelFile, FileMode.Open);
            FileStream ifsLabels = new FileStream(labelFile, FileMode.Open);
            BinaryReader brImages = new BinaryReader(ifsPixels);
            BinaryReader brLabels = new BinaryReader(ifsLabels);
            int magic1 = brImages.ReadInt32(); // stored as big endian
            magic1 = ReverseBytes(magic1); // convert to Intel format
            int imageCount = brImages.ReadInt32();
            imageCount = ReverseBytes(imageCount);
            int numRows = brImages.ReadInt32();
            numRows = ReverseBytes(numRows);
            int numCols = brImages.ReadInt32();
            numCols = ReverseBytes(numCols);
            int magic2 = brLabels.ReadInt32();
            magic2 = ReverseBytes(magic2);
            int numLabels = brLabels.ReadInt32();
            numLabels = ReverseBytes(numLabels);
            for (int di = 0; di < numImages; ++di)
            {
                for (int i = 0; i < 28; ++i) // get 28x28 pixel values
                {
                    for (int j = 0; j < 28; ++j)
                    {
                        byte b = brImages.ReadByte();
                        pixels[i][j] = b;
                    }
                }
                byte lbl = brLabels.ReadByte(); // get the label
                DigitImage dImage = new DigitImage(28, 28, pixels, lbl);
                result[di] = dImage;
            } // Each image
            ifsPixels.Close(); brImages.Close();
            ifsLabels.Close(); brLabels.Close();
            return result;
        }
        public static int ReverseBytes(int v)
        {
            byte[] intAsBytes = BitConverter.GetBytes(v);
            Array.Reverse(intAsBytes);
            return BitConverter.ToInt32(intAsBytes, 0);
        }
        public static Bitmap MakeBitmap(DigitImage dImage, int mag)
        {
            int width = dImage.width * mag;
            int height = dImage.height * mag;
            Bitmap result = new Bitmap(width, height);
            Graphics gr = Graphics.FromImage(result);
            for (int i = 0; i < dImage.height; ++i)
            {
                for (int j = 0; j < dImage.width; ++j)
                {
                    int pixelColor = 255 - dImage.pixels[i][j]; // black digits
                    System.Drawing.Color c = System.Drawing.Color.FromArgb(pixelColor, pixelColor, pixelColor);
                    SolidBrush sb = new SolidBrush(c);
                    gr.FillRectangle(sb, j * mag, i * mag, mag, mag);
                }
            }
            return result;
        }
        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            DigitImage currImage = trainImages[0];
            Bitmap bitMap = MakeBitmap(currImage, 6);
            foto.Source = bitMap.ToBitmapImage();
        }
        private void Button_Click_2(object sender, RoutedEventArgs e)
        {
               Task<bool> LernTask =  Lern();  
        }

        public async Task<bool> Lern()
        {
            List<Task> tasks = new List<Task>();
            await Task.Factory.StartNew(async () =>
            {

                await GenerateRandomWeight();
                await GenerateBias();
                await LoadTreningData(trainImages);

        
                //for every image
                for (int c = 0; c < 60000; c++)
                {
                    DigitImage currImage = trainImages[c];
                    Bitmap bitMap = MakeBitmap(currImage, 6);
                    foto.Dispatcher.Invoke(new UpdateImageCallback(this.UpdateImage), bitMap);
                    await CountHLOutput(c);
                    await CountOLOutput(c);


                    await ErrorTotals(c);

                    await ErrorHiddenOuT(c);
                    await ErrorHiddenIn(c);
                }

                await LoadTreningData(testImages);
                for (int t = 0; t < 10000; t++)
                {
                    await CountHLOutput(t);
                    await CountOLOutput(t);
                    await CheckResult(t);
                }
            });
            return true;
        }
        public Task LoadTreningData(DigitImage[] Images)
        {
            int imageNum = 0;
            foreach (DigitImage image in Images)
            {

                int pixel = 0;
                for (int inputW = 0; inputW < 28; inputW++)
                {
                    for (int inputH = 0; inputH < 28; inputH++)
                    {
                        ImagesPixels[imageNum,pixel] = image.pixels[inputW][inputH]/25;
                        pixel++;
                    }
                }
                labels[imageNum] = image.label;
                imageNum++;
            }
            return Task.CompletedTask;
        }
        public Task GenerateRandomWeight()
        {
            Random rnd = new Random();
            for (int p = 0; p < 784; p++)
            {
                for (int h = 0; h < hiddenLayer1.Length; h++)
                {
                    weighth[p, h] = 2 * rnd.NextDouble() -1;
                }
            }

            for (int h = 0; h < hiddenLayer1.Length; h++)
            {
                for (int o = 0; o < outputs.Length; o++)
                {
                    weighto[h, o] = 2 * rnd.NextDouble() - 1;
                }
            }
            return Task.CompletedTask;
        }
        public Task GenerateBias()
        {
            for(int bh = 0; bh < biash.Length; bh++)
            {
                biash[bh] = 3.5;
            }
            for(int bo = 0; bo < biaso.Length; bo++)
            {
                biaso[bo] = 1.2;
            }
            return Task.CompletedTask;
        }
        public Task CountHLOutput(int ImageNum)
        {
            for (int p = 0; p < pixelCount; p++)
            {
                double input = ImagesPixels[ImageNum, p];
                for (int h = 0; h < hiddenLayer1.Length; h++)
                {
                    OutH1[h] += weighth[p, h] * input;
                }
            }
            for (int b = 0; b< biash.Length; b++)
            {
                OutH1[b] -= biash[b];
            }
            int counter = 0;
            foreach (double Out in OutH1)
            {
                //Debug.WriteLine(Math.Pow(Math.E, Out));
                OutH1[counter] = 1 / (1 - Math.Pow(Math.E, Out * -1));
                counter++;
            }
            return Task.CompletedTask;
        }
        public Task CountOLOutput(int ImageNum)
        {
            for (int OOut = 0; OOut < OutH1.Length; OOut++)
            {
                double input = OutH1[OOut];
                for (int o = 0; o < outputs.Length; o++)
                {
                    outputs[o] += weighto[OOut, o] * input;
                }
            }
            for (int b = 0; b < biaso.Length; b++)
            {
                outputs[b] -= biaso[b];
            }
            int counter = 0;
            foreach (double Out in outputs)
            { 
                //Debug.WriteLine($"{counter}-{Math.Pow(Math.E, Out)}-{labels[ImageNum]}");
                outputs[counter] = 1 / (1 - Math.Pow(Math.E, Out * -1));
                counter++;
            }
            return Task.CompletedTask;
        }
        public Task ErrorTotals(int ImageNum)
        {
            ErrorTotalOutputs = 0;
            for (int o = 0; o < outputs.Length; o++)
            {
                ErrorTotalOutputs += Math.Pow((labels[ImageNum] - outputs[o]), 2) / 2;
                outputBox.Dispatcher.Invoke(
                new UpdateTextCallback(this.UpdateText),
                new object[] { $"{o}-{Math.Pow((labels[ImageNum] - outputs[o]), 2) / 2}-{labels[ImageNum]}"});
            }
            //outputBox.AppendText($"{o}-{Math.Pow((labels[ImageNum] - outputs[o]), 2) / 2}-{labels[ImageNum]}");
            //$"{o}-{Math.Pow((labels[ImageNum] - outputs[o]), 2) / 2}-{labels[ImageNum]}"
            //await OutputMessageToLogWindow("Calkowity:");
            //await OutputMessageToLogWindow(ErrorTotalOutputs.ToString());
            return Task.CompletedTask;

        }
        public Task CheckResult(int ImageNum)
        {
            ErrorTotalOutputs = 0;
            double[] result = new double[10];
            for (int o = 0; o < outputs.Length; o++)
            {
                result[o] = Math.Pow((labels[ImageNum] - outputs[o]), 2) / 2;
                outputBox.Dispatcher.Invoke(
            new UpdateTextCallback(this.UpdateText),
            new object[] { $"{o}-{result[o]}-{labels[ImageNum]}" });
          
            }
            return Task.CompletedTask;
        }
        public Task ErrorHiddenOuT(int ImageNum)
        {
            for (int l = 0; l < outputs.Length; l++)
            {
                EtotOuts[l] = -(labels[ImageNum] - outputs[l]);///eo1outo1
                OutNets[l] = outputs[l] * (1 - outputs[l]);///outo1neto1
                for (int h = 0; h < OutH1.Length; h++)
                {
                    EtotNets[l, h] = OutH1[h] * EtotOuts[l] * OutNets[l];
                    weighto[h, l] -= LernRate * EtotNets[l, h];
                   //Debug.WriteLine($"Changes n out:{LernRate * EtotNets[l, h]}");
                }
                
            }
            return Task.CompletedTask;
        }
        public Task ErrorHiddenIn(int ImageNum)
        {
            for (int i = 0; i < pixelCount; i++)
            {
                double input = ImagesPixels[ImageNum, i];
                for (int h = 0; h < OutH1.Length; h++)
                {
                    OutNetH[h] += OutH1[h] * (1 - OutH1[h]);
                    for (int l = 0; l < outputs.Length; l++)
                    {
                        ETotalOut[h] += EtotNets[l, h];
                    }
                    weighth[i,h] -= (-1 * LernRate * (OutNetH[h] * ETotalOut[h]  * input));
                    //Debug.WriteLine($"Changes on in:{-1 * LernRate * (OutNetH[h] * ETotalOut[h] * input)}");
                    
                    OutNetH[h] = 0;
                    ETotalOut[h] = 0;
                }
            }
            return Task.CompletedTask;
        }
        private static readonly object synchLock = new object();

        private void OutputMessageToLogWindow(string message)
        {
                    this.outputBox.AppendText(message);
        }

        private async void Button_Click_3(object sender, RoutedEventArgs e)
        {
            outputBox.AppendText("ok");
            List<Task> list = new List<Task>();
            await Task.Factory.StartNew(() =>
            {
                do
                {
                    someAsync();
         
                } while (true);
            });
            outputBox.AppendText("ok2");
        }
        public Task someAsync()
        {
            
            return Task.CompletedTask;
        }
        private void UpdateText(string message)
        {
            outputBox.AppendText(message + "\n");
            outputBox.AppendText("\u2028"); // Linebreak, not paragraph break
            outputBox.ScrollToEnd();
        }
        private void UpdateImage(Bitmap image)
        {
            foto.Source = image.ToBitmapImage();
        }
     }
}

