This is a project to create a model optimizer system that optimize a Neural Network structure to get high scores

The planned stages for the project:
1. Define a robust sql diagram that allowes to store all needed data of the system
2. Create an sql server to hold the data and be called upon by the future server
3. Create a server that manages testing, having a schedular to decide what tests to send next when requested
4. Create code to generate new tests that runs on the server, probably using openai api for that
5. Create code to run a test and returns a result, for now only tested using CIFAR-10, and in future would probably need some major changes
6. Create 2 codes options to act as clients:
   a. http client, runs without stop, send request to server for a test, runs it, return result, asks for the next, has a full working console menu, for all options
   b. manual, gets the place of a file with tests to run, return json results
   
 The big plan is to create a container that has the server, the uri of the sql server, and the dataset, and then allowing to upload that container into a service such as google cloud run, 
and have clients from diffrent pcs connect to the server and running the tests, and even having the ability to run a manual client on top of a gpu cluster with a big test request



Current design for the sql database:
![SiteDataDiagram-DL database drawio(11)](https://github.com/user-attachments/assets/e88f290e-5dfa-48cd-a04e-22976e738900)
