# These are extra models i am working on but the RAG Model is complete
        # PDF_IntegratedwithLLM- this will give a short to the point answer
        # self.LocalLLM_PDF()

        # this is still under construction
        # self.LocalLLM_PDF_Fallback()


# _______________________________________________SEMANTIC SEARCH____________________________________________________
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
    query = "symptoms of pellagra"
    while True:
        query = input("Enter your query for Semantic Search (type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Exiting Semantic Search.")
            break
        # Get just the scores and indices of top related results
        scores, indices = self.retrieve_relevant_resources(query=query)
        print(scores, indices)

        # Print out the texts of the top scores
        self.print_top_results_and_scores(query=query)


# _______________________________________________________LocaL LLM Model____________________________________________
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
    print("________________________________________________________________________________________")
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
        query = input("Enter your query for Semantic Search (type 'exit' to quit): ").strip()
        print(f"Query Being Searched :\n{query}")
        if query.lower() == "exit":
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


# ______________________________________________PDF+LLM_____________________________________________________________
def prompt_formatter(self, query: str, context_items: list[dict]) -> str:
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

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
                        \nNow, using only the information provided in the context items below, answer the following user query as clearly and accurately as possible.
                        {context}
                        \nUser query: {query}
                        Answer:"""
    return base_prompt.format(context=context, query=query)


def fallback_prompt(self, query: str) -> str:
    return f"Please answer the following question as clearly and accurately as possible:\n\nQuestion: {query}\nAnswer:"


def AddingPDFDatatoLLMModel(self, query, top_k=5, max_new_tokens=256):
    # print(f"\n[INFO] Running RAG-powered query: {query}\n")
    # scores, indices = self.retrieve_relevant_resources(query=query)
    # context_items = [self.pages_and_chunks[i] for i in indices]
    # prompt = self.prompt_formatter(query=query, context_items=context_items)
    # tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.model_id)
    # inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    # output = self.llm_model.generate(**inputs, max_new_tokens=512)
    # response = tokenizer.decode(output[0], skip_special_tokens=True)
    # print("Model Response:\n", response)

    scores, indices = self.retrieve_relevant_resources(query=query)
    # Collect the context chunks
    context_items = [self.pages_and_chunks[i] for i in indices]
    # Create a combined prompt
    prompt = self.prompt_formatter(query=query, context_items=context_items)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.model_id)
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = self.llm_model.generate(
        **input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True
    )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n[MODEL RESPONSE]")
    print(output_text.replace(prompt, "").strip())


# Modify LocalLLM method to call the RAG method
def LocalLLM_PDF_Fallback(self):
    if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"

    model_id = self.model_id

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
    self.llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        torch_dtype=torch.float16,
        quantization_config=self.quantization_config if self.use_quantization_config else None,
        low_cpu_mem_usage=False,
        attn_implementation=attn_implementation
    )

    if not self.use_quantization_config:
        self.llm_model.to("cuda")

    self.get_model_num_params(self.llm_model)
    self.get_model_mem_size(self.llm_model)

    while True:
        query = input("Enter your query for Semantic RAG-powered LLM (type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Exiting LLM session.")
            break

        self.AddingPDFDatatoLLMModel(query)


# ________________________________________LLM +PDF Fallback_________________________________________________________

def LLM_Model_Feature_Prompt(self):
    while True:
        query = input("Enter your query for Semantic Search (type 'exit' to quit): ").strip()
        print(f"Query Being Searched :\n{query}")
        if query.lower() == "exit":
            print("Exiting Semantic Search.")
            break

        scores, indices = self.retrieve_relevant_resources(query=query)
        print(scores, indices)
        # Create a list of context items
        context_items = [self.pages_and_chunks[i] for i in indices]

        # Format prompt with context items
        prompt = self.prompt_formatter_LLM(query=query,
                                           context_items=context_items)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(prompt)


def prompt_formatter_LLM(self, query: str,
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
                        \nNow, using only the information provided in the context items below, answer the following user query as clearly and accurately as possible.
                        {context}
                        \nUser query: {query}
                        Answer:"""

    # Update base prompt with context items and query
    base_prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
         "content": base_prompt}
    ]

    # Apply the chat template
    model_id = self.model_id
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return prompt
