from ImportsForRag import *

# Config class for ingestion
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list

config = DataIngestionConfig(
    root_dir=Path("Dataset"),
    source_URL="https://raw.githubusercontent.com/furkhansuhail/ProjectData/main/RagModel_Bedrock_Data/human-nutrition-text.pdf",
    local_data_file=Path("Dataset/human-nutrition-text.pdf"),
    STATUS_FILE="Dataset/status.txt",
    ALL_REQUIRED_FILES=[]
)

def download_project_file(source_URL, local_data_file):
    local_data_file.parent.mkdir(parents=True, exist_ok=True)
    if local_data_file.exists():
        print(f"✅ File already exists at: {local_data_file}")
    else:
        print(f"⬇ Downloading file from {source_URL}...")
        file_path, _ = request.urlretrieve(url=source_URL, filename=local_data_file)
        print(f"✅ File downloaded and saved to: {file_path}")

# download_project_file(config.source_URL, config.local_data_file)

class RagModel:
    def __init__(self):
        # Downloading the pdf file from github
        download_project_file(config.source_URL, config.local_data_file)

        # defining objects
        self.Dataset = pd.DataFrame()
        self.statsDataset = pd.DataFrame()
        self.pdf_path = pdf_path
        self.device = device
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2", device=self.device)
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                      bnb_4bit_compute_dtype=torch.float16)
        # self.model_id = "google/gemma-2b-it"
        self.model_id = "google/gemma-7b-it"
        self.use_quantization_config = True
        self.ModelDriver()

    def ModelDriver(self):
        # Reading the dataset
        self.pages_and_text_list = self.ReadingPDF()

        # Convert the pdf to a dataframe object
        self.statsDataset = pd.DataFrame(self.pages_and_text_list)

        # Sentencizing the Data
        self.Sentencizing_NLP()

        # Chunking
        self.Chunking()

        # Splitting Chunks
        # Splitting each chunk into its own item
        self.SplittingChunks()

        # Run once then comment it out so that we get embeddings saved to a csv file
        self.EmbeddingChunks()

        # Searching for the results of a query in pdf without LLM
        # THis is done by using the dot product between the vecotrs
        self.Semantic_Rag_DotProduct_Search()

        # Semantic Search - Functionalizing Semantic Search
        self.SemanticSearch()

        # Local LLM Model
        # self.LocalLLM()


        # Local LLM Fallback if query not found in pdf
        # self.PromptFeature_LLM()



    def text_formatter(self, text: str) -> str:
        """Performs minor formatting on text."""
        cleaned_text = text.replace("\n",
                                    " ").strip()  # note: this might be different for each doc (best to experiment)

        # Other potential text formatting functions can go here
        return cleaned_text

    def ReadingPDF(self):
        doc = fitz.open(self.pdf_path)  # open a document
        pages_and_text = []
        for page_number, page in tqdm(enumerate(doc)):  # iterate the document pages
            text = page.get_text()  # get plain text encoded as UTF-8
            text = self.text_formatter(text)
            pages_and_text.append(
                {"page_number": page_number - 41,  # adjust page numbers since our PDF starts on page 42
                 "page_char_count": len(text),
                 "page_word_count": len(text.split(" ")),
                 "page_sentence_count_raw": len(text.split(". ")),
                 "page_token_count": len(text) / 4,
                 # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                 "text": text})
        return pages_and_text

    def Sentencizing_NLP(self):
        nlp = English()

        # Add a sentencizer pipeline, see https://spacy.io/api/sentencizer/
        nlp.add_pipe("sentencizer")

        for item in tqdm(self.pages_and_text_list):
            item["sentences"] = list(nlp(item["text"]).sents)

            # Make sure all sentences are strings
            item["sentences"] = [str(sentence) for sentence in item["sentences"]]

            # Count the sentences
            item["page_sentence_count_spacy"] = len(item["sentences"])

        # print(random.sample(self.pages_and_text_list, k=1))
        self.statsDataset = pd.DataFrame(self.pages_and_text_list)
        # print(self.statsDataset.describe().round(2))

    def Chunking(self):
        # Define split size to turn groups of sentences into chunks
        num_sentence_chunk_size = 10

        # Create a function that recursively splits a list into desired sizes
        def split_list(input_list: list,
                       slice_size: int) -> list[list[str]]:
            """
            Splits the input_list into sublists of size slice_size (or as close as possible).

            For example, a list of 17 sentences would be split into two lists of [[10], [7]]
            """
            return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

        # Loop through pages and texts and split sentences into chunks
        for item in tqdm(self.pages_and_text_list):
            item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                                 slice_size=num_sentence_chunk_size)
            item["num_chunks"] = len(item["sentence_chunks"])

        # Sample an example from the group (note: many samples have only 1 chunk as they have <=10 sentences total)

        # Create a DataFrame to get stats
        # print(random.sample(self.pages_and_text_list, k=1))
        self.statsDataset = pd.DataFrame(self.pages_and_text_list)
        # print(self.statsDataset.describe().round(2))

    def SplittingChunks(self):
        # Split each chunk into its own item
        self.pages_and_chunks = []
        for item in tqdm(self.pages_and_text_list):
            for sentence_chunk in item["sentence_chunks"]:
                chunk_dict = {}
                chunk_dict["page_number"] = item["page_number"]

                # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
                joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1',
                                               joined_sentence_chunk)  # ".A" -> ". A" for any full-stop/capital letter combo
                chunk_dict["sentence_chunk"] = joined_sentence_chunk

                # Get stats about the chunk
                chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
                chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4  # 1 token = ~4 characters

                self.pages_and_chunks.append(chunk_dict)

        # How many chunks do we have?
        # print(len(self.pages_and_chunks))
        # Get stats about our chunks
        self.statsDataset = pd.DataFrame(self.pages_and_chunks)
        # print(self.statsDataset.describe().round(2))

        # Show random chunks with under 30 tokens in length
        min_token_length = 30
        for row in self.statsDataset[self.statsDataset["chunk_token_count"] <=
                                     min_token_length].sample(5).iterrows():
            print(f'Chunk token count: {row[1]["chunk_token_count"]} | Text: {row[1]["sentence_chunk"]}')

        self.pages_and_chunks_over_min_token_len = self.statsDataset[
            self.statsDataset["chunk_token_count"] > min_token_length].to_dict(orient="records")
        # print(self.pages_and_chunks_over_min_token_len[:2])

    def EmbeddingChunks(self):

        # Send the model to the GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # requires a GPU installed, for reference on my local machine, I'm using a NVIDIA RTX 2080
        self.embedding_model.to(device)

        # Create embeddings one by one on the GPU
        for item in tqdm(self.pages_and_chunks_over_min_token_len):
            item["embedding"] = self.embedding_model.encode(item["sentence_chunk"])

        # Turn text chunks into a single list
        text_chunks = [item["sentence_chunk"] for item in self.pages_and_chunks_over_min_token_len]

        # Embed all texts in batches
        # text_chunk_embeddings = self.embedding_model.encode(text_chunks,
        #                                                     batch_size=32,
        #                                                     convert_to_tensor=True)
        # print(text_chunk_embeddings)

        # Embed all texts in batches
        self.embeddings = self.embedding_model.encode(text_chunks,
                                                      batch_size=32,
                                                      convert_to_tensor=True).to(self.device)
        # Save to pickle
        embeddings_save_path = "text_chunks_and_embeddings.pkl"
        with open(embeddings_save_path, "wb") as f:
            pickle.dump(self.pages_and_chunks_over_min_token_len, f)
        print(f"✅ Embeddings saved to {embeddings_save_path}")

    #_________________________________________Phase 1 Dot Product  Search in PDF________________________________________

    def Semantic_Rag_DotProduct_Search(self):
        text_chunks_and_embedding_df = None
        # Import texts and embedding df
        with open("text_chunks_and_embeddings.pkl", "rb") as f:
            data = pickle.load(f)
            text_chunks_and_embedding_df = pd.DataFrame(data)

        # Convert embedding column back to np.array (it got converted to string when it got saved to CSV)
        # text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
        #     lambda x: np.fromstring(x.strip("[]"), sep=" "))

        text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
            lambda x: np.fromstring(x.strip("[]"), sep=" ") if isinstance(x, str) else x)

        # Convert texts and embedding df to list of dicts
        self.pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

        # Convert embeddings to torch tensor and send to device
        # (note: NumPy arrays are float64, torch tensors are float32 by default)
        self.embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()),
                                       dtype=torch.float32).to(
            self.device)
        # print(self.embeddings.shape)
        # print(text_chunks_and_embedding_df.head())
        # print(self.embeddings[0].dtype)
        self.SearchQuery()

    def SearchQuery(self):
        query = "macronutrients functions"
        print(f"Query: {query}")
        PageNumberList = []

        # 2. Embed the query to the same numerical space as the text examples
        # Note: It's important to embed your query with the same model you embedded your examples with.
        # query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True).to(self.device)
        # 3. Get similarity scores with the dot product (we'll time this for fun)
        start_time = timer()
        dot_scores = util.dot_score(a=query_embedding, b=self.embeddings)[0]
        end_time = timer()

        print(f"Time take to get scores on {len(self.embeddings)} embeddings: {end_time - start_time:.5f} seconds.")

        # 4. Get the top-k results (we'll keep this to 5)
        top_results_dot_product = torch.topk(dot_scores, k=5)
        print(top_results_dot_product)
        print(f"Query: '{query}'\n")
        print("Results:")
        # Loop through zipped together scores and indicies from torch.topk
        for score, idx in zip(top_results_dot_product[0], top_results_dot_product[1]):
            print(f"Score: {score:.4f}")
            # Print relevant sentence chunk (since the scores are in descending order,
            # the most relevant chunk will be first)
            print("Text:")
            self.print_wrapped(self.pages_and_chunks[idx]["sentence_chunk"])
            # Print the page number too so we can reference the textbook further (and check the results)
            print(f"Page number: {self.pages_and_chunks[idx]['page_number']}")
            PageNumberList.append(self.pages_and_chunks[idx]['page_number'])
            print("\n")
        self.CheckResult_PDF_Page(query, PageNumberList)

    def CheckResult_PDF_Page(self, query, PageNumberList):
        for pageNumber in PageNumberList:
            # Open PDF and load target page
            pdf_path = self.pdf_path  # requires PDF to be downloaded
            doc = fitz.open(pdf_path)
            page = doc.load_page(pageNumber + 41)  # number of page (our doc starts page numbers on page 41)

            # Get the image of the page
            img = page.get_pixmap(dpi=300)

            # Optional: save the image
            # img.save("output_filename.png")
            doc.close()

            # Convert the Pixmap to a numpy array
            img_array = np.frombuffer(img.samples_mv,
                                      dtype=np.uint8).reshape((img.h, img.w, img.n))

            # Display the image using Matplotlib
            plt.figure(figsize=(13, 10))
            plt.imshow(img_array)
            plt.title(f"Query: '{query}' | Most relevant page:")
            plt.axis('off')  # Turn off axis
            plt.show()

    def print_wrapped(self, text, wrap_length=80):
        wrapped_text = textwrap.fill(text, wrap_length)
        print(wrapped_text)


    # ______________________________________Semantic Search Pipeline____________________________________________________

    def retrieve_relevant_resources(self, query: str,
                                    n_resources_to_return: int = 5,
                                    print_time: bool = True):
        model = self.embedding_model
        # Embed the query
        query_embedding = model.encode(query, convert_to_tensor=True)

        # Get dot product scores on embeddings
        start_time = timer()
        dot_scores = util.dot_score(query_embedding, self.embeddings)[0]
        end_time = timer()

        if print_time:
            print(
                f"[INFO] Time taken to get scores on {len(self.embeddings)} embeddings: {end_time - start_time:.5f} seconds.")

        scores, indices = torch.topk(input=dot_scores,
                                     k=n_resources_to_return)

        return scores, indices

    def print_top_results_and_scores(self, query):
        """
        Takes a query, retrieves most relevant resources and prints them out in descending order.

        Note: Requires pages_and_chunks to be formatted in a specific way (see above for reference).
        """

        pages_and_chunks = self.pages_and_chunks
        n_resources_to_return: int = 5

        scores, indices = self.retrieve_relevant_resources(query=query)

        print(f"Query: {query}\n")
        print("Results:")
        # Loop through zipped together scores and indicies
        for score, index in zip(scores, indices):
            print(f"Score: {score:.4f}")
            # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
            self.print_wrapped(pages_and_chunks[index]["sentence_chunk"])
            # Print the page number too so we can reference the textbook further and check the results
            print(f"Page number: {pages_and_chunks[index]['page_number']}")
            print("\n")

    def SemanticSearch(self):
        while True:
            query = input("Enter your query for Semantic Search (type 'LLM' to move to next Function or exit to quit): ").strip()
            if query.lower() == "llm":
                self.LocalLLM()
            elif query.lower() == "exit":
                print("Exiting Semantic Search.")
                break
            # Get just the scores and indices of top related results
            scores, indices = self.retrieve_relevant_resources(query=query)
            # Print out the texts of the top scores
            self.print_top_results_and_scores(query=query)

    #___________________________________________________________________________________________________________________

                                        # Adding LLM Model for Local Generation with Prompt Feature

    #___________________________________________________________________________________________________________________
    def get_model_num_params(self, model: torch.nn.Module):
        return sum([param.numel() for param in model.parameters()])

    def get_model_mem_size(self, model: torch.nn.Module):
        """
        Get how much memory a PyTorch model takes up.

        See: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822
        """
        # Get model parameters and buffer sizes
        mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
        mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

        # Calculate various model sizes
        model_mem_bytes = mem_params + mem_buffers  # in bytes
        model_mem_mb = model_mem_bytes / (1024 ** 2)  # in megabytes
        model_mem_gb = model_mem_bytes / (1024 ** 3)  # in gigabytes

        return {"model_mem_bytes": model_mem_bytes,
                "model_mem_mb": round(model_mem_mb, 2),
                "model_mem_gb": round(model_mem_gb, 2)}

    def LocalLLM(self):
        # Bonus: Setup Flash Attention 2 for faster inference, default to "sdpa" or "scaled dot product attention" if it's not available
        # Flash Attention 2 requires NVIDIA GPU compute capability of 8.0 or above, see: https://developer.nvidia.com/cuda-gpus
        # Requires !pip install flash-attn, see: https://github.com/Dao-AILab/flash-attention
        if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "sdpa"

        # print(f"[INFO] Using attention implementation: {attn_implementation}")

        # 2. Pick a model we'd like to use (this will depend on how much GPU memory you have available)
        # model_id = "google/gemma-7b-it"
        model_id = self.model_id  # (we already set this above)
        # print(f"[INFO] Using model_id: {model_id}")

        # 3. Instantiate tokenizer (tokenizer turns text into numbers ready for the model)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
        # 4. Instantiate the model
        self.llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id,
                                                              torch_dtype=torch.float16,
                                                              # datatype to use, we want float16
                                                              quantization_config=self.quantization_config if self.use_quantization_config else None,
                                                              low_cpu_mem_usage=False,  # use full memory
                                                              attn_implementation=attn_implementation)  # which attention version to use

        if not self.use_quantization_config:  # quantization takes care of device setting automatically, so if it's not used, send model to GPU
            self.llm_model.to("cuda")

        # print(llm_model)
        self.get_model_num_params(self.llm_model)
        self.get_model_mem_size(self.llm_model)

        while True:
            query = input("Enter your query for Semantic Search (type 'llm_prompt' to quit): ").strip()
            print(f"Query Being Searched :\n{query}")
            if query.lower() == "llm_prompt":
                self.PromptFeature_LLM()
            elif query.lower() == "exit":
                print("Exiting Semantic Search.")
                break

            # Create prompt template for instruction-tuned model
            dialogue_template = [
                {"role": "user",
                 "content": query}
            ]

            # Apply the chat template
            prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                                   tokenize=False,  # keep as raw text (not tokenized)
                                                   add_generation_prompt=True)
            # print(f"\nPrompt (formatted):\n{prompt}")

            # Tokenize the input text (turn it into numbers) and send it to GPU
            input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
            # print(f"Model input (tokenized):\n{input_ids}\n")

            # Generate outputs passed on the tokenized input
            # See generate docs: https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/text_generation#transformers.GenerationConfig
            outputs = self.llm_model.generate(**input_ids,
                                              max_new_tokens=256)  # define the maximum number of new tokens to create
            # print(f"Model output (tokens):\n{outputs[0]}\n")

            outputs_decoded = tokenizer.decode(outputs[0])
            # print(f"Model output (decoded):\n{outputs_decoded}\n")

            # print(f"Input text: {input_text}\n")
            print(f"Output text:\n{outputs_decoded.replace(prompt, '').replace('<bos>', '').replace('<eos>', '')}")

    # ___________________________________________________________________________________________________________________

                                        # Prompt Feature Feature

    # ___________________________________________________________________________________________________________________
    def prompt_formatter(self, query: str,
                         context_items: list[dict]) -> str:
        """
        Augments query with text-based context from context_items.
        """
        # Join context items into one dotted paragraph
        context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

        # Create a base prompt with examples to help the model
        # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
        # We could also write this in a txt file and import it in if we wanted.
        base_prompt = """Based on the following context items, please answer the query. 
                        Give yourself room to think by extracting relevant passages from the context before answering the query. 
                        Don't return the thinking, only return the answer. Make sure your answers are as explanatory as possible. 
                        Use the following examples as reference for the ideal answer style. 
                        
                        \nExample 1: 
                        Query: What are the fat-soluble vitamins? 
                        Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K... 
                        
                        \nExample 2:
                        Query: What are the causes of type 2 diabetes? 
                        Answer: Type 2 diabetes is often associated with overnutrition... 
                        
                        \nExample 3: 
                        Query: What is the importance of hydration for physical performance? 
                        Answer: Hydration is crucial for physical performance... 
                        \nContext: 
                        \nNow, using only the information provided in the context items below, answer the following user query as clearly and accurately as possible.
                        \n
                        \n
                        \n
                        _____________________________________________________________________________________________
                        User query: {query}
                        Answer:
                        \n
                        {context}
                        \n
                        _____________________________________________________________________________________________
                        """

        # Update base prompt with context items and query
        base_prompt = base_prompt.format(context=context, query=query)

        # Create prompt template for instruction-tuned model
        dialogue_template = [
            {"role": "user",
             "content": base_prompt}
        ]

        # 2. Pick a model we'd like to use (this will depend on how much GPU memory you have available)
        # model_id = "google/gemma-7b-it"
        model_id = self.model_id  # (we already set this above)
        # 3. Instantiate tokenizer (tokenizer turns text into numbers ready for the model)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)

        # Apply the chat template
        prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                               tokenize=False,
                                               add_generation_prompt=True)
        return prompt

    def PromptFeature_LLM(self):
        while True:
            query = input("Enter your query for Semantic Search (type pdf for next phase or 'exit' to quit): ").strip()
            print(f"Query Being Searched :\n{query}")
            if query.lower() == "pdf":
                self.PPD_LLM()
            elif query.lower() == "exit":
                print("Exiting Semantic Search.")
                break

            # Get just the scores and indices of top related results
            scores, indices = self.retrieve_relevant_resources(query=query)

            # Create a list of context items
            context_items = [self.pages_and_chunks[i] for i in indices]

            # Format prompt with context items
            prompt = self.prompt_formatter(query=query,
                                      context_items=context_items)
            print(prompt)

    # ___________________________________________________________________________________________________________________

    # PDF + LLM

    # ___________________________________________________________________________________________________________________

    def ask(self, query,
            temperature=0.7,
            max_new_tokens=512,
            format_answer_text=True,
            return_answer_only=True):
        """
        Takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.
        """

        # Get just the scores and indices of top related results
        scores, indices = self.retrieve_relevant_resources(query=query)

        # Create a list of context items
        context_items = [self.pages_and_chunks[i] for i in indices]

        # Add score to context item
        for i, item in enumerate(context_items):
            item["score"] = scores[i].cpu()  # return score back to CPU

        # Format the prompt with context items
        prompt = self.prompt_formatter(query=query,
                                  context_items=context_items)

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.model_id)

        # Tokenize the prompt
        input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

        # Generate an output of tokens
        outputs = self.llm_model.generate(**input_ids,
                                     temperature=temperature,
                                     do_sample=True,
                                     max_new_tokens=max_new_tokens)

        # Turn the output tokens into text
        output_text = tokenizer.decode(outputs[0])

        if format_answer_text:
            # Replace special tokens and unnecessary help message
            output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace(
                "Sure, here is the answer to the user query:\n\n", "")

        # Only return the answer without the context items
        if return_answer_only:
            return output_text

        return output_text, context_items


    def PPD_LLM(self):
        while True:
            query = input("Enter your query for Semantic Search (type 'exit' to quit): ").strip()
            print(f"Query Being Searched :\n{query}")
            if query.lower() == "exit":
                print("Exiting Semantic Search.")
                break
            # Answer query with context and return context
            answer, context_items = self.ask(query=query, temperature=0.7, max_new_tokens=512, return_answer_only=False)

            print(f"Answer:\n")
            self.print_wrapped(answer)
            print(f"Context items:")
            # context_items

RagModelObj = RagModel()