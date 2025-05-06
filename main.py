from flask import Flask, request, jsonify, render_template
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import time

# Load environment variables from the .env file
load_dotenv("secret.env")

# Access the API key from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check if the API key is available
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is missing")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test', methods=['GET'])
def test():
    return "Flask API is working!"

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/prompts')
def prompts():
    return render_template('prompts.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        start_time = time.time()
        if not request.is_json:
            return jsonify({'error': 'Request must be in JSON format'}), 400

        data = request.get_json()
        prompt_input = data.get('prompt')
        tone = data.get('tone', 'informative')
        audience = data.get('audience', 'general')
        length = data.get('length', 'concise')

        if not prompt_input:
            return jsonify({'error': 'Prompt cannot be empty'}), 400

        modified_prompt = f"Generate a {tone} article for a {audience} audience with a {length} length on the topic: {prompt_input}. Make sure to provide an apt title(highlighted in bold) and adhere to the mentioned style."

        prompt_template = PromptTemplate.from_template(modified_prompt)
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
        chain = LLMChain(llm=llm, prompt=prompt_template)

        output = chain.run({"title": prompt_input})
        response_time = time.time() - start_time
        return jsonify({'output': output, 'response_time': response_time})
    
    except Exception as e:
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

@app.route('/ask_query', methods=['POST'])
def ask_query():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be in JSON format'}), 400

        data = request.get_json()
        query = data.get('query')
        generated_content = data.get('generated_content', '')

        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400

        if not generated_content:
            query_prompt = f"The user has a question: {query}. Answer it to the best of your ability. Understand the context from this question itself"
        else:
            query_prompt = f"Answer the user's question, taking context from the article provided below. \nQuestion: {query}\nIf the answer to the question is present in the article, display it. If the answer is not present in the article then answer relevantly to the best of your ability, understanding the context. Search the web if required:\n\nArticle: {generated_content}.\n\n"
        
        prompt_template = PromptTemplate.from_template(query_prompt)
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
        chain = LLMChain(llm=llm, prompt=prompt_template)

        answer = chain.run({})
        return jsonify({'answer': answer})
    
    except Exception as e:
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be in JSON format'}), 400

        data = request.get_json()
        content = data.get('content')
        
        if not content:
            return jsonify({'error': 'Content cannot be empty'}), 400

        evaluation_prompt = f"""Evaluate this article content on the following parameters by giving appropriate scores.
        Just provide the scores against parameters and nothimg more.:
        1. Response time(provide the time you(Gemini Flah 2.0 LLM) took to generate the article)
        2. Clarity and coherence
        3. Grammar and spelling
        4. Structure and flow
        5. Tone consistency
        6. Factual Accuracy
        7. Plagiarism
        8. Overall quality
        
        Content to evaluate:
        {content}
        
        Use Google fact check API for factual accuracy any free tool for plagiarism detection(PlagiarismCheck.org API). Provide your evaluation in bullet points bolding the scores."""

        prompt_template = PromptTemplate.from_template(evaluation_prompt)
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
        chain = LLMChain(llm=llm, prompt=prompt_template)

        results = chain.run({})
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)