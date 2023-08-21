import gradio as gr
import openai
import numpy as np
from time import sleep
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader

COMPLETIONS_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Initialize the OpenAI API
openai.api_key = "YOUR_OPEN_AI_KEY"


# Function to convert a PDF to text
def extract_text_from_pdf(pdf_file, progress=gr.Progress()):

    try:
        reader = UnstructuredPDFLoader(pdf_file.name)
        data = reader.load()
        text = data[0].page_content

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            length_function=len,
        )
        chunks = text_splitter.create_documents([text])

        embed = compute_doc_embeddings(chunks, progress)
        return chunks, embed, "uploaded"
    except:
        return None, None, ""


def get_embedding(text, model=EMBEDDING_MODEL):
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    return result["data"][0]["embedding"]


def compute_doc_embeddings(text, progress):
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    result = {}
    for idx in progress.tqdm(range(len(text))):
        try:
            res = get_embedding(text[idx].page_content)
        except:
            done = False
            while not done:
                sleep(5)
                try:
                    res = get_embedding(text[idx].page_content)
                    done = True
                except:
                    pass
        result[idx] = res

    return result


def vector_similarity(x, y):
    """
    Returns the similarity between two vectors.

    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(query, contexts):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities


SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))
COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}


def construct_prompt(question, context_embeddings, df):
    """
    Fetch relevant
    """
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)

    if "email" in question:
        MAX_SECTION_LEN = 2500
        COMPLETIONS_API_PARAMS['max_tokens'] = 1000
        COMPLETIONS_API_PARAMS['temperature'] = 0.5
        header = """Write email using the provided context \n\nContext:\n """
    elif "summary" in question or "summarize" in question:
        MAX_SECTION_LEN = 2500
        COMPLETIONS_API_PARAMS['max_tokens'] = 1000
        COMPLETIONS_API_PARAMS['temperature'] = 0.5
        header = """Write detailed summary of the provided context \n\nContext:\n """
        question = ""
    else:
        MAX_SECTION_LEN = 1000
        COMPLETIONS_API_PARAMS['max_tokens'] = 300
        COMPLETIONS_API_PARAMS['temperature'] = 0.0
        header = """Answer the question in detail as truthfully as possible, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n """

    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        document_section = df[section_index].page_content
        chosen_sections_len += len(document_section) * 0.25 + separator_len

        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))

    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))

    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"


def answer_query_with_context(
        query,
        df,
        document_embeddings, history,
        openchat, show_prompt=True
):
    history = history or []
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )

    if show_prompt:
        print(prompt)
    openchat = openchat or [{"role": "system", "content": "You are a Q&A assistant"}]
    openchat.append({"role": "user", "content": prompt})
    response = openai.ChatCompletion.create(
        messages=openchat,
        **COMPLETIONS_API_PARAMS
    )
    openchat.pop()
    openchat.append({"role": "user", "content": query})
    print(COMPLETIONS_API_PARAMS)
    output = response["choices"][0]["message"]["content"].replace('\n', '<br>')
    openchat.append({"role": "assistant", "content": output})
    history.append((query, output))
    return history, history, openchat, ""


with gr.Blocks() as app:
    history_state = gr.State()
    document = gr.Variable()
    embeddings = gr.Variable()
    chat = gr.Variable()
    with gr.Row():
        upload = gr.File(label=None, interactive=True, elem_id="short-upload-box")
        ext = gr.Textbox(label="Progress")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot().style(color_map=("#075e54", "grey"))

    with gr.Row():
        message = gr.Textbox(label="What's on your mind??",
                             placeholder="What's the answer to life, the universe, and everything?",
                             lines=1)
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    upload.change(extract_text_from_pdf, inputs=[upload], outputs=[document, embeddings, ext])
    message.submit(answer_query_with_context, inputs=[message, document, embeddings, history_state, chat],
                   outputs=[chatbot, history_state, chat, message])
    submit.click(answer_query_with_context, inputs=[message, document, embeddings, history_state, chat],
                 outputs=[chatbot, history_state, chat, message])
if __name__ == "__main__":
    app.queue().launch(server_name="0.0.0.0", debug=True)