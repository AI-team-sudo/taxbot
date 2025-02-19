import streamlit as st
from pinecone import Pinecone
from openai import OpenAI
from typing import List
from deep_translator import GoogleTranslator
import time
import math

# Debug mode
DEBUG = st.sidebar.checkbox("Debug Mode", False)

# System prompt definition
system_prompt = """You are an authoritative expert on the Gujrat Tax Law and the Ahmedabad Municipal Corporation.
Your responses should be:
1. Comprehensive and detailed
2. Include step-by-step procedures when applicable
3. Quote relevant sections directly from the Tax Act
4. Provide specific references (section numbers, chapters, and page numbers)
5. Break down complex processes into numbered steps
6. Include any relevant timelines or deadlines
7. Mention any prerequisites or requirements
8. Highlight important caveats or exceptions

For every fact or statement, include a reference to the source document and page number in this format:
[Source: Document_Name, Page X]

Always structure your responses in a clear, organized manner using:
- Bullet points for lists
- Numbered steps for procedures
- Bold text for important points
- Separate sections with clear headings"""

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize Pinecone
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index("gujtaxlaw")

def chunk_text(text: str, max_tokens: int = 3000) -> List[str]:
    """Split text into chunks of approximately max_tokens."""
    # Rough approximation: 1 token ≈ 4 characters
    chars_per_chunk = max_tokens * 4

    # Split by newlines first to maintain context
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        if current_length + paragraph_length > chars_per_chunk and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [paragraph]
            current_length = paragraph_length
        else:
            current_chunk.append(paragraph)
            current_length += paragraph_length

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks

def get_embedding(text: str) -> List[float]:
    """Get embedding for the input text using OpenAI's embedding model."""
    # First translate Gujarati text to English if needed
    if any(ord(c) >= 0x0A80 and ord(c) <= 0x0AFF for c in text):
        try:
            text = translate_text(text, 'en')
        except Exception as e:
            if DEBUG:
                st.error(f"Translation error in get_embedding: {str(e)}")
            pass

    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def search_pinecone(query: str, k: int = 3):
    """Search Pinecone index with embedded query."""
    query_embedding = get_embedding(query)
    results = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True
    )
    return results

def translate_text(text: str, target_lang: str) -> str:
    """Translate text to target language using deep-translator with error handling."""
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)

        # Add retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Split text into smaller chunks if it's too long
                max_chunk_size = 4000  # Google Translate limit
                if len(text) > max_chunk_size:
                    chunks = [text[i:i+max_chunk_size]
                            for i in range(0, len(text), max_chunk_size)]
                    translated_chunks = []
                    for chunk in chunks:
                        translated_chunk = translator.translate(chunk)
                        translated_chunks.append(translated_chunk)
                    return ' '.join(translated_chunks)
                else:
                    return translator.translate(text)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)  # Wait before retrying

    except Exception as e:
        raise Exception(f"Translation failed: {str(e)}")

def generate_response(query: str, context: str, system_prompt: str):
    """Generate response using OpenAI with context and system prompt, handling long contexts."""
    # Split context into manageable chunks
    context_chunks = chunk_text(context)

    # Initialize response parts
    response_parts = []

    # Process each chunk
    for i, chunk in enumerate(context_chunks):
        # Modify the prompt based on chunk position
        if i == 0:
            chunk_prompt = f"Part 1/{len(context_chunks)} of the context. Answer based on this and upcoming parts: {query}"
        else:
            chunk_prompt = f"Part {i+1}/{len(context_chunks)} of the context. Continue building the answer with this additional information."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {chunk}\n\nQuestion: {chunk_prompt}"}
        ]

        # Add previous response parts for context if not first chunk
        if i > 0:
            messages.append({"role": "assistant", "content": "Previous parts of the response: " + " ".join(response_parts)})

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )

            response_parts.append(response.choices[0].message.content)

            # Add a small delay between API calls to avoid rate limits
            time.sleep(1)

        except Exception as e:
            if DEBUG:
                st.error(f"Error generating response for chunk {i+1}: {str(e)}")
            continue

    # Combine all response parts
    final_response = "\n\n".join(response_parts)

    # Generate a summary/consolidation if there were multiple chunks
    if len(response_parts) > 1:
        try:
            summary_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Original question: {query}\n\nPlease provide a concise, coherent summary of the following detailed response: {final_response}"}
            ]

            summary_response = client.chat.completions.create(
                model="gpt-4",
                messages=summary_messages,
                temperature=0.7,
                max_tokens=1000
            )

            final_response = summary_response.choices[0].message.content

        except Exception as e:
            if DEBUG:
                st.error(f"Error generating summary: {str(e)}")

    return final_response

# Streamlit UI
st.title("ગુજરાત કર કાયદો સહાયક | Gujarat Tax Law Assistant")
st.write("કર કાયદા વિશે કોઈપણ પ્રશ્ન પૂછો | Ask any question about the Tax Law")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and processing
if prompt := st.chat_input("તમે શું જાણવા માંગો છો? | What would you like to know?"):
    # Check if input is in Gujarati and translate to English if needed
    if any(ord(c) >= 0x0A80 and ord(c) <= 0x0AFF for c in prompt):
        with st.spinner('Translating your question...'):
            english_prompt = translate_text(prompt, 'en')
            # Display original Gujarati query and its English translation
            with st.chat_message("user"):
                st.markdown(f"Original Query (ગુજરાતી): {prompt}")
                st.markdown(f"Translated Query (English): {english_prompt}")
            st.session_state.messages.append({"role": "user", "content": prompt})
            prompt = english_prompt
    else:
        # Display original English query
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Search Pinecone and get relevant context
    with st.spinner('Searching knowledge base...'):
        search_results = search_pinecone(prompt)
        context = "\n".join([result.metadata.get('text', '') for result in search_results.matches])

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner('Generating response...'):
            response = generate_response(prompt, context, system_prompt)

        # Create container for response and translation
        response_container = st.container()

        with response_container:
            # Display original response
            st.markdown("### English Response:")
            st.markdown(response)

            # Create a key for managing translation state
            translation_key = f"translate_{len(st.session_state.messages)}"

            # Initialize translation state if not exists
            if f"translated_{translation_key}" not in st.session_state:
                st.session_state[f"translated_{translation_key}"] = False

            # Translation button
            if st.button("ગુજરાતી માં વાંચો 🔄", key=translation_key):
                st.session_state[f"translated_{translation_key}"] = True

            # Show translation if button was clicked
            if st.session_state[f"translated_{translation_key}"]:
                try:
                    with st.spinner('Translating to Gujarati...'):
                        gujarati_response = translate_text(response, 'gu')
                        st.markdown("### ગુજરાતી અનુવાદ:")
                        st.markdown(gujarati_response)

                        if DEBUG:
                            st.write("Debug Info:")
                            st.write(f"Response length: {len(response)}")
                            st.write(f"Translation attempt for key: {translation_key}")
                except Exception as e:
                    st.error(f"Translation failed: {str(e)}")
                    st.error("Please try again or contact support if the issue persists.")

        st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with additional information
with st.sidebar:
    st.header("વિશે | About")
    st.write("""
    આ ચેટબોટ ગુજરાત કર કાયદા અને અમદાવાદ મ્યુનિસિપલ કોર્પોરેશન વિશે માહિતી પ્રદાન કરે છે.

    This chatbot provides information about the Gujarat Tax Law and Ahmedabad Municipal Corporation.
    """)
    st.write("""
    ભાષા સુવિધાઓ | Language Features:
    - તમે ગુજરાતી અથવા અંગ્રેજીમાં પ્રશ્નો પૂછી શકો છો
    - જો તમે ગુજરાતીમાં પૂછશો, તો તે આપોઆપ અંગ્રેજીમાં અનુવાદ થશે
    - ગુજરાતીમાં જવાબો જોવા માટે 'ગુજરાતી માં વાંચો 🔄' બટન પર ક્લિક કરો

    - You can ask questions in English or Gujarati
    - If you ask in Gujarati, it will be automatically translated to English
    - Click the 'ગુજરાતી માં વાંચો 🔄' button to see responses in Gujarati
    """)
