# Name: Swaraj Bhanja | Student Id: st125052

# Welcome to Language Model!

This is a web-based end-to-end application named Language Model. It leverages the power of deep learning and web development to provide a website that returns a 50 word context based on the input word.

# About the Deep Learning Model

The brains of this solution is the deep learning model trained for this purpose. The DL model was trained based on the HuggingFace **Newsgroup** Atheism dataset. The complete **Newsgroup** dataset is derived from the classic 20 Newsgroups dataset, a collection of approximately 20000 newsgroup documents organized into 20 different categories. The subset focusing on **Atheism** includes documents, which contains discussions and debates about atheism, religion, and related topics. This dataset is particularly useful for experimenting with NLP models in scenarios that involve topic-specific or sensitive content. **LSTM** technique was used to train the model using this dataset.


# Loading the Dataset

A list of dataset configurations available in **Newsgroup** is created first. Each configuration represents various topics such as atheism, computer graphics, etc. However, only one configuration, '18828_alt.atheism', is active in the current setup, while the rest are commented out.

Next, an empty list to store individual datasets is created. For each active configuration, the script loads the dataset associated with the active configuration. In this case, the only active cconfiguration here is '18828_alt.atheism'. Hence, only this dataset gets loaded. Only the training portion of each dataset ('train') is retrieved and appended to the empty list. This is the only portion currently present in the dataset.

Once all selected datasets are loaded, they are combined into a single dataset. This results in a combined dataset containing examples from all the individual active datasets specified in configs. Due to lack of compute resources and time, only one dataset is present in the combined dataset. However, the main aim to establish the idea of combining multiple datasets can be seen from this logic.

After combining the datasets, the script splits the data into three parts: training, validation, and testing. First, the splitting function divides the combined dataset into two portions: 80% for training and 20% for temporary testing. The temporary testing set is then further split equally into validation and final testing subsets, ensuring that each receives 10% of the original data.

Finally, the three subsets: training, validation, and testing are organized into a dataset dictionary. This structure provides a convenient way to access the different splits by their respective keys ('train', 'validation', and 'test'), facilitating further use in model training and evaluation.

# Tokenization

Tokens can be thought of as individual words or meaningful segments of text. A tokenizer is initialized using the torchtext library. The tokenizer is configured to process text using basic English rules. This includes converting text to lowercase and splitting it into words based on spaces and punctuation. Next, a lambda function takes in a single example of data and the tokenizer as inputs. It generates a list of tokens and is stored as an output. The lambda function is applied to every example in the dataset. The result is a new dataset, where each example has been transformed to include tokens.

# Numericalization

This process involves converting tokens into numbers forming a vocabulary. This vocabulary is essentially a mapping between tokens and unique integers associated with them. The function ensures that only tokens appearing at least three times in the dataset are included in the vocabulary. Rare tokens that occur fewer times are excluded, which helps reduce noise and the size of the vocabulary. To handle special cases, two special tokens are added to the vocabulary:
> - `unk` (index 0): This token represents any unknown word that is not present in the vocabulary.
> - `eos` (index 1): This token is used to mark the end of a sequence, helping models understand when a sentence or input ends.
The `unk` token is also used to ensure that if any word is not found in the vocabulary, the model can handle it gracefully instead of failing with some absurd error. You could relate it with the Graceful Recovery design patterns used in Software Engineering.

# Preparing Data Using Batch Loader

The tokenized dataset is split into train, test and validation sets using a batch loader approach with the batch size as 128. This is essential since the DL model must use the train and validation datasets to ensure that it learns effectively instead of memorizing.

# About the LSTMLanguageModel class

The LSTMLanguageModel class is a neural network designed to predict the next word in a sequence using an LSTM architecture. It transforms tokens into dense vectors via an embedding layer, processes these vectors with stacked LSTM layers to capture temporal relationships, and predicts the next word through a fully connected layer. Dropout is applied to prevent overfitting, and careful weight initialization ensures stable training. Hidden and cell states are initialized as zero tensors to maintain information across sequences. The detach_hidden method prevents backpropagation across sequence boundaries, optimizing memory usage. During the forward pass, tokenized inputs are embedded, processed by the LSTM, and passed through the fully connected layer to generate probabilities over the vocabulary. This design enables the model to learn complex patterns in text, ensuring effective sequence modeling and robust predictions while minimizing overfitting.

# Training

First, the model is initialized with specific dimensions: emb_dim and hid_dim represent the sizes of the embedding and hidden states, respectively, while num_layers determines the depth of the LSTM. The dropout rate is set to 0.65 to possibly prevent overfitting, and the learning rate for the optimizer is 1e-3. The Adam optimizer is used for efficient parameter updates, and the loss function is CrossEntropyLoss, appropriate for multi-class classification.  The get_batch function prepares input (src) and target (target) sequences from the data. For each batch, src is a slice of tokens, while target is the same slice shifted by one token to represent the next word prediction. The train function manages the training loop. Before each epoch, the hidden states of the LSTM are reset to ensure the model doesn't carry information across epochs. The data is adjusted to ensure its length is compatible with the sequence length (seq_len), ensuring consistency across batches. Within each iteration, the model processes a batch of data. Gradients are reset at the start, and the hidden states are detached from the computation graph to improve efficiency. The model's predictions (prediction) and the actual targets (target) are compared using the loss function, which measures how well the model predicts the next word in the sequence. The loss is backpropagated to compute gradients, and the gradients are clipped to prevent instability during training. Finally, the optimizer updates the model’s parameters to minimize the loss. 50 epochs spanning around 4 hours were invoked. A `ReduceLROnPlateau` learning scheduler was also used which ensures that if the loss doesn't improve in a certain epoch, the learning rate is decreased by a specified factor for more granular fitting. The aim is to ensure that the model can learn effectively instead of memorizing.

## Testing
For testing purposes, the generate function takes a prompt, converts it into token indices using the tokenizer and vocabulary, and sends it into the model as an input. The model predicts the next token in an autoregressive manner, using its output as input for the next step's input. A temperature parameter controls the randomness of predictions: lower temperatures produce focused outputs, while higher temperatures create more varied but potentially less coherent text. The function stops generating when it reaches the desired sequence length or encounters an end-of-sequence `(eos)` token, whichever is achieved earlier.

## Pickling The Model
The LSMT DL model was chosen for deployment.
> The pickled model was saved using a .pkl extension to be used later in a web-based application

# Website Creation
The model was then hosted over the Internet with Flask as the backend, HTML, CSS, JS as the front end, and Docker as the container. The following describes the key points of the hosting discussion.
> **1. DigitalOcean (Hosting Provider)**
> 
>> - **Role:** Hosting and Server Management
>> - **Droplet:** Hosts the website on a virtual server, where all files, databases, and applications reside.
>> - **Dockerized Container:** The website is hosted in a Dockerized container running on the droplet. The container is built over a Ubuntu Linux 24.10 image.
>> - **Ports and Flask App:** The Dockerized container is configured to host the website on port 8000. It forwards requests to port 5000, where the Flask app serves the backend and static files. This flask app contains the pickled model, which is used for prediction.
>> - **IP Address:** The droplet’s public IP address directs traffic to the server.
>
>  **In Summary:** DigitalOcean is responsible for hosting the website within a Dockerized container, ensuring it is online and accessible via its IP address.
> 
>  **2. GoDaddy (Domain Registrar)**
>
>> - **Role:** Domain Registration and Management
>> - **Domain Purchase:** Registers and manages the domain name.
>> - **DNS Management:** Initially provided DNS setup, allowing the domain to be pointed to the DigitalOcean droplet’s IP address.
> 
> **In Summary:** GoDaddy ensures the domain name is registered and correctly points to the website’s hosting server.
>
>  **3. Cloudflare (DNS and Security/Performance Optimization)**
>
>> - **Role:** DNS Management, Security, and Performance Optimization
>> - **DNS Management:** Resolves the domain to the correct IP address, directing traffic to the DigitalOcean droplet.
>> - **CDN and Security:** Caches website content globally, enhances performance, and provides security features like DDoS protection and SSL encryption.
> 
> **In Summary:** Cloudflare improves the website’s speed, security, and reliability.
>
> **How It Works Together:**
> 
>> - **Domain Resolution:** The domain is registered with GoDaddy, which points it to Cloudflare's DNS servers. Cloudflare resolves the domain to the DigitalOcean droplet's IP address.
>> - **Content Delivery:** Cloudflare may serve cached content or forward requests to DigitalOcean, which processes and serves the website content to users.
> 
> **Advantages of This Setup:**
>
>> - **Security:** Cloudflare provides DDoS protection, SSL/TLS encryption, and a web application firewall.
>> - **Performance:** Cloudflare’s CDN reduces load times by caching content globally, while DigitalOcean offers scalable hosting resources.
>> - **Reliability:** The combination of GoDaddy, Cloudflare, and DigitalOcean ensures the website is always accessible, with optimized DNS resolution and robust hosting.


# Access The Final Website
You can access the website [here](https://aitmltask.online). 

# Limitations
Note that currently, the solution supports slightly meaningful context generation only on a limited words related to Atheism like "Atheism is", "Atheism is good ", etc. The model may generate gibberish for certain inputs and is a known limitation.


# How to Run the Search Engine Docker Container Locally
### Step 1: Clone the Repository
> - First, clone the repository to your local machine.
### Step 2: Install Docker
> - If you don't have Docker installed, you can download and install it from the [Docker](https://www.docker.com) website.
### Step 3: Build and Run the Docker Container
Once Docker is installed, navigate to the app folder in the project directory. Delete the docker-compose-deployment.yml file and run the following commands to build and run the Docker container:
> - `docker compose up -d`

### Important Notes
> - The above commands will serve the Docker container on port 5000 and forward the requests to the Flask application running on port 5000 in the containerized environment.
> - Ensure Ports Are Free: Make sure that port 5000 is not already in use on your machine before running the container.
> - Changing Flask's Port: If you wish to change the port Flask runs on (currently set to 5000), you must update the port in the app.py file. After making the change, remember to rebuild the Docker image in the next step. Execute the following command to stop the process: `docker compose down`. Then goto Docker Desktop and delete the container and image from docker. 

