This is a project to create a model optimizer system that optimize a Neural Network structure to get high scores

The planned stages for the project:
1. Define a robust sql diagram that allowes to store all needed data of the system
2. Create an sql server to hold the data and be called upon by the future server
3. Create a server that manages testing, having a schedular to decide what tests to send next when requested
4. Create code to generate new tests that runs on the server, probably using openai api for that
5. Create code to run a test and returns a result
6. Create 2 codes options to act as clients:
   a. http client, runs without stop, send request to server for a test, runs it, return result, asks for the next
   b. manual, recieves a string of tests to be done, returns string of results

 The big plan is to create a container that has the server, the uri of the sql server, and the dataset, and then allowing to upload that container into a service such as google cloud run, 
and have clients from diffrent pcs connect to the server and running the tests, and even having the ability to run a manual client on top of a gpu cluster with a big test request

![SiteDataDiagram-DL database drawio(2)](https://github.com/user-attachments/assets/5ce8542e-8eed-4e5b-bdcf-93889bbf5ed7)

