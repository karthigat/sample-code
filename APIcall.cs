using System;
//using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace x911_Alert
{
    class Program
    {
        static void Main(string[] args)
        {
            //string Url, Location, value1, Description, value2, Servitiy, value3;
           
            MakeHttpRequest makeHttp = new MakeHttpRequest();

            makeHttp.RequestAPI1("https://hackathon.xmatters.com/api/integration/1/functions/fc37f92d-33c0-4763-8d8d-43bf890a729e/triggers?apiKey=85079475-e03e-46ae-b4f6-c6a693461e80", "json", "Location", "Site1", "Description", "Emergency", "Severity", "High", "Scale", "100");

        }
    }
    class MakeHttpRequest
    {
        public void RequestAPI1(string url, string type, string param1, string param1value, string param2, string param2value, string param3, string param3value, string param4, string param4value)
        {

            try
            {
                var httpWebRequest = (HttpWebRequest)WebRequest.Create(url);
                httpWebRequest.ContentType = "application/json";
                httpWebRequest.Method = "POST";
                using (var streamWriter = new

                StreamWriter(httpWebRequest.GetRequestStream()))
                {
                    string json = "{\"" + param1 + "\": \"" + param1value + "\",\"" + param2 + "\":\"" + param2value + "\",\"" + param3 + "\":\"" + param3value + "\",\"" + param4 + "\":\"" + param4value + "\"}";

                    streamWriter.Write(json);
                }

                var httpResponse = (HttpWebResponse)httpWebRequest.GetResponse();
                if (httpResponse.StatusCode == HttpStatusCode.Accepted)
                {
                    using (var streamReader = new StreamReader(httpResponse.GetResponseStream()))
                    {
                        var result = streamReader.ReadToEnd();
                        // amt = JsonConvert.DeserializeObject<cymax>(result).amt;

                    }
                }

            }
            catch (Exception e)
            {

            }
        }
    }
    }
