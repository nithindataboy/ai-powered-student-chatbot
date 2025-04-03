from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import torch
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from googleapiclient.discovery import build
from dotenv import load_dotenv
from fuzzywuzzy import process
import google.generativeai as genai  
import base64
from io import BytesIO
import requests
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss
from bs4 import BeautifulSoup
import re
from fuzzywuzzy import fuzz
from urllib.parse import quote_plus
import time
import logging
from functools import lru_cache
import concurrent.futures

app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# API Keys
API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  

# Validate API Keys
def validate_api_keys():
    missing_keys = []
    if not API_KEY:
        missing_keys.append("GOOGLE_API_KEY")
    if not CSE_ID:
        missing_keys.append("GOOGLE_CSE_ID")
    if not GEMINI_API_KEY:
        missing_keys.append("GEMINI_API_KEY")
    
    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Gemini AI
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro")
except Exception as e:
    logger.error(f"Failed to configure Gemini AI: {str(e)}")
    model = None

# Initialize BERT model and tokenizer
try:
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
except Exception as e:
    logger.error(f"Failed to initialize BERT model: {str(e)}")
    bert_tokenizer = None
    bert_model = None

# Initialize sentence transformer for better semantic search
try:
    sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    logger.error(f"Failed to initialize sentence transformer: {str(e)}")
    sentence_transformer = None

# Predefined QA pairs
predefined_qa_pairs = {
    "admission process": "Admissions at HITS are through *EAMCET, ECET, and Management Quota*. The process involves entrance exam scores, document verification, and counseling.",
    "chairman of HITS": "The chairman of HITS is *Dr. Siddarth Reddy Arminanda*, known for his contributions to education and institutional development.",
    "departments available": "HITS offers *CSE, AI/ML, IoT, Mechanical, Civil, ECE, etc.* with modern labs, research centers, and highly qualified faculty.",
    "placements": "Top recruiters include *TCS, Infosys, Wipro, Cognizant, and Capgemini*. The college provides placement training, internships, and career guidance.",
    "documents for admission": "Required documents: *10th & 12th marksheets, TC, Entrance Score, Aadhar, Passport-size photos, and Category Certificate (if applicable).*",
    "courses offered": "HITS offers a variety of programs including *B.Tech* in fields like Artificial Intelligence, Civil Engineering, Computer Science, and more. *M.Tech* programs and *MBA* specializations are also available.",
    "eligibility criteria": "Eligibility varies by program. For *B.Tech*, candidates must have completed 10+2 with relevant subjects and qualifying entrance exams. Detailed criteria are available on the official website.",
    "fee structure": "The fee structure differs across programs and categories. For the most accurate and up-to-date information, please refer to the *Fee Structure* section on the HITS website.",
    "scholarships available": "HITS provides various scholarships based on merit and other criteria. Prospective students are encouraged to check the *Scholarships* section on the website for detailed information.",
    "admission deadlines": "Admission deadlines vary annually. It's advisable to consult the *Admissions* section of the HITS website or contact the admissions office directly for the current academic year's deadlines.",
    "placement opportunities": "HITS has a dedicated *Career Development Center* that facilitates placements with top recruiters such as TCS, Infosys, and Wipro. The center also offers training and internship opportunities.",
    "hostel facilities": "Separate hostel facilities are available for both boys and girls, equipped with necessary amenities to ensure a comfortable stay for students.",
    "transportation services": "HITS offers transportation services across various routes in Hyderabad, ensuring safe and convenient travel for students and staff.",
    "campus infrastructure": "The campus boasts modern infrastructure, including well-equipped labs, libraries, sports facilities, and classrooms designed to enhance the learning experience.",
    "international collaborations": "HITS has established tie-ups with international universities, providing students with opportunities for exchange programs and global exposure.",
    "anti-ragging policy": "HITS maintains a strict *Anti-Ragging & Disciplinary Committee* to ensure a safe and conducive environment for all students.",
    "grievance redressal": "Students can address their concerns through the *Online Grievance Redressal* system available on the HITS website, ensuring timely resolution of issues.",
    "research programs": "The institute encourages research through various *Research Programmes & Workshops*, fostering innovation and academic growth among students and faculty.",
    "industry tie-ups": "HITS has collaborations with industries like *Microsoft, IBM, and Oracle*, providing students with exposure to current technologies and industry practices.",
    "extracurricular activities": "A range of extracurricular activities, including sports, cultural events, and technical clubs, are available to support the holistic development of students."
}

def load_qa_pairs():
    """Load QA pairs from file if exists, otherwise use predefined pairs"""
    try:
        if os.path.exists('qa_pairs.json'):
            with open('qa_pairs.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading QA pairs: {str(e)}")
    return predefined_qa_pairs

def save_qa_pairs(qa_pairs):
    """Save QA pairs to file"""
    try:
        with open('qa_pairs.json', 'w') as f:
            json.dump(qa_pairs, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving QA pairs: {str(e)}")

@lru_cache(maxsize=100)
def get_gemini_suggestion(query):
    """Get AI suggestion using Gemini model with caching"""
    try:
        if not model:
            return None
            
        prompt = f"""As an AI assistant for HITS College, please provide a helpful response to this query: {query}
        Keep the response concise, accurate, and relevant to HITS College information."""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Gemini AI error: {str(e)}")
        return None

def get_bert_embedding(text):
    """Get BERT embedding for text with error handling"""
    try:
        if not bert_tokenizer or not bert_model:
            return None
            
        inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    except Exception as e:
        logger.error(f"BERT embedding error: {str(e)}")
        return None

def get_semantic_similarity(query, text):
    """Get semantic similarity using sentence transformer"""
    try:
        if not sentence_transformer:
            return 0
            
        query_embedding = sentence_transformer.encode([query])[0]
        text_embedding = sentence_transformer.encode([text])[0]
        return cosine_similarity([query_embedding], [text_embedding])[0][0]
    except Exception as e:
        logger.error(f"Semantic similarity error: {str(e)}")
        return 0

def get_faq_answer_with_bert(question):
    """Get FAQ answer using BERT and semantic search"""
    try:
        if not bert_model or not sentence_transformer:
            return None, 0
            
        best_match = None
        best_score = 0
        
        # Get query embedding
        query_embedding = get_bert_embedding(question)
        if query_embedding is None:
            return None, 0
            
        # Compare with all FAQ questions
        for q, a in predefined_qa_pairs.items():
            # Get question embedding
            q_embedding = get_bert_embedding(q)
            if q_embedding is None:
                continue
                
            # Calculate cosine similarity
            similarity = cosine_similarity([query_embedding], [q_embedding])[0][0]
            
            # Also get semantic similarity
            semantic_score = get_semantic_similarity(question, q)
            
            # Combine scores
            combined_score = (similarity + semantic_score) / 2
            
            if combined_score > best_score:
                best_score = combined_score
                best_match = (q, a)
        
        return best_match, best_score
    except Exception as e:
        logger.error(f"BERT FAQ search error: {str(e)}")
        return None, 0

def get_alternative_search_results(query):
    """Alternative search function using multiple search engines"""
    try:
        # URL encode the query
        encoded_query = quote_plus(query)
        results = []
        
        # Try DuckDuckGo API first
        try:
            ddg_url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json"
            ddg_response = requests.get(ddg_url)
            
            if ddg_response.status_code == 200:
                ddg_data = ddg_response.json()
                
                # Add abstract if available
                if ddg_data.get('Abstract'):
                    results.append({
                        'title': ddg_data.get('Heading', ''),
                        'link': ddg_data.get('AbstractURL', ''),
                        'snippet': ddg_data.get('Abstract', ''),
                        'type': 'text',
                        'image_url': None
                    })
                
                # Add related topics
                for topic in ddg_data.get('RelatedTopics', [])[:3]:
                    if 'Text' in topic:
                        results.append({
                            'title': topic.get('FirstURL', '').split('/')[-1].replace('_', ' '),
                            'link': topic.get('FirstURL', ''),
                            'snippet': topic.get('Text', ''),
                            'type': 'text',
                            'image_url': None
                        })
        except Exception as e:
            print(f"DuckDuckGo search failed: {str(e)}")
        
        # Try Bing Web Search API
        try:
            bing_url = f"https://api.bing.microsoft.com/v7.0/search"
            headers = {
                'Ocp-Apim-Subscription-Key': os.getenv('BING_API_KEY', ''),
                'Accept': 'application/json'
            }
            params = {
                'q': query,
                'count': 5,
                'responseFilter': 'Webpages,Images'
            }
            
            bing_response = requests.get(bing_url, headers=headers, params=params)
            
            if bing_response.status_code == 200:
                bing_data = bing_response.json()
                
                # Add web results
                for result in bing_data.get('webPages', {}).get('value', [])[:3]:
                    results.append({
                        'title': result.get('name', ''),
                        'link': result.get('url', ''),
                        'snippet': result.get('snippet', ''),
                        'type': 'text',
                        'image_url': None
                    })
                
                # Add image results
                for image in bing_data.get('images', {}).get('value', [])[:2]:
                    results.append({
                        'title': image.get('name', ''),
                        'link': image.get('hostPageUrl', ''),
                        'snippet': '',
                        'type': 'image',
                        'image_url': image.get('contentUrl', '')
                    })
        except Exception as e:
            print(f"Bing search failed: {str(e)}")
        
        # If still no results, try web scraping as last resort
        if not results:
            try:
                # Try HITS website first
                hits_url = f"https://www.hits.ac.in/search?q={encoded_query}"
                hits_response = requests.get(hits_url, headers={'User-Agent': 'Mozilla/5.0'})
                
                if hits_response.status_code == 200:
                    soup = BeautifulSoup(hits_response.text, 'html.parser')
                    
                    # Find all search results
                    for item in soup.find_all(['div', 'article'], class_=['search-result', 'post']):
                        title = item.find(['h2', 'h3', 'h4'])
                        link = item.find('a')
                        snippet = item.find(['p', 'div'], class_=['snippet', 'excerpt'])
                        
                        if title and link:
                            results.append({
                                'title': title.get_text().strip(),
                                'link': link.get('href', ''),
                                'snippet': snippet.get_text().strip() if snippet else '',
                                'type': 'text',
                                'image_url': None
                            })
                
                # If no HITS results, try general web search
                if not results:
                    search_url = f"https://www.google.com/search?q={encoded_query}"
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    response = requests.get(search_url, headers=headers)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Find text results
                        for div in soup.find_all('div', class_='g'):
                            title_elem = div.find('h3')
                            link_elem = div.find('a')
                            snippet_elem = div.find('div', class_='VwiC3b')
                            
                            if title_elem and link_elem:
                                results.append({
                                    'title': title_elem.get_text().strip(),
                                    'link': link_elem.get('href', ''),
                                    'snippet': snippet_elem.get_text().strip() if snippet_elem else '',
                                    'type': 'text',
                                    'image_url': None
                                })
                        
                        # Find image results
                        for img in soup.find_all('img', class_='rg_i'):
                            parent = img.find_parent('a')
                            if parent:
                                results.append({
                                    'title': img.get('alt', 'Image'),
                                    'link': parent.get('href', ''),
                                    'snippet': '',
                                    'type': 'image',
                                    'image_url': img.get('src', '')
                                })
            except Exception as e:
                print(f"Web scraping failed: {str(e)}")
        
        # If still no results, return a default response
        if not results:
            results = [{
                'title': 'No direct results found',
                'link': '#',
                'snippet': 'Try rephrasing your query or check the FAQ section for related information.',
                'type': 'text',
                'image_url': None
            }]
        
        return results[:5]  # Return top 5 results
        
    except Exception as e:
        print(f"Alternative search error: {str(e)}")
        return [{
            'title': 'Search Error',
            'link': '#',
            'snippet': 'An error occurred while searching. Please try again.',
            'type': 'text',
            'image_url': None
        }]

def google_search(query, num_results=5):
    """Enhanced Google search function with fallback to alternative search"""
    try:
        if not API_KEY or not CSE_ID:
            print("API keys are missing")
            return get_alternative_search_results(query)

        service = build("customsearch", "v1", developerKey=API_KEY)
        
        # First try with site restriction
        try:
            result = service.cse().list(
                q=query,
                cx=CSE_ID,
                num=num_results,
                siteSearch="hits.ac.in"
            ).execute()
        except Exception as e:
            print(f"Site-restricted search failed: {str(e)}")
            return get_alternative_search_results(query)

        # If no results, try without site restriction
        if not result or 'items' not in result or not result['items']:
            try:
                result = service.cse().list(
                    q=query,
                    cx=CSE_ID,
                    num=num_results
                ).execute()
            except Exception as e:
                print(f"General search failed: {str(e)}")
                return get_alternative_search_results(query)

        # Process results
        if result and 'items' in result:
            processed_results = []
            for item in result['items']:
                # Clean up title
                title = item.get('title', '')
                if ' - HITS' in title:
                    title = title.split(' - HITS')[0].strip()
                
                # Get image URL if available
                image_url = None
                if 'pagemap' in item and 'cse_image' in item['pagemap']:
                    image_url = item['pagemap']['cse_image'][0].get('src')
                elif 'image' in item:
                    image_url = item['image'].get('contextLink')
                
                # Determine result type
                result_type = 'image' if image_url else 'text'
                
                # Create result object
                result_obj = {
                    'title': title,
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'type': result_type,
                    'image_url': image_url
                }
                
                # Only add if we have at least a title or snippet
                if result_obj['title'] or result_obj['snippet']:
                    processed_results.append(result_obj)
            
            # If we have results, return them
            if processed_results:
                print(f"Found {len(processed_results)} results")
                return processed_results
            
            # If no processed results, try alternative search
            return get_alternative_search_results(query)
            
        print("No results found")
        return get_alternative_search_results(query)
        
    except Exception as e:
        print(f"Error in google_search: {str(e)}")
        return get_alternative_search_results(query)

# 🔥 **Find Best Match for Short Queries**
def find_best_match(query, documents):
    """Find best matching document for a query"""
    best_match, score = process.extractOne(query, documents.keys())
    return best_match if score > 70 else None

# 🔥 **Semantic Search for Best Context**
def semantic_search(query, documents):
    """Perform semantic search to find best matching document"""
    best_match = find_best_match(query, documents)
    return documents[best_match] if best_match else None

# 🔥 **BERT Model for Contextual Answers**
def get_bert_answer(question, context):
    """Get contextual answer using BERT model"""
    try:
        inputs = bert_tokenizer.encode_plus(
            question, context, add_special_tokens=True, return_tensors="pt",
            padding=True, truncation=True, max_length=512
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            outputs = bert_model(input_ids, attention_mask=attention_mask)

        start_scores, end_scores = outputs.start_logits, outputs.end_logits
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)

        if start_index >= end_index:
            return None

        answer_tokens = input_ids[0][start_index:end_index+1]
        return bert_tokenizer.decode(answer_tokens, skip_special_tokens=True)
    except Exception as e:
        print(f"Error in get_bert_answer: {str(e)}")
        return None

# 🔥 **Convert AI Answer to Speech**
def speak_text(text):
    """Convert text to speech"""
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error in speak_text: {str(e)}")

def stop_speech():
    """Stop text-to-speech"""
    try:
        engine = pyttsx3.init()
        engine.stop()
    except Exception as e:
        print(f"Error in stop_speech: {str(e)}")

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    """Enhanced search endpoint with better error handling and response formatting"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Please provide a query'}), 400

        # First try to get FAQ answer
        faq_answer = get_faq_answer(query)
        
        # Then get search results
        search_results = google_search(query)
        
        # If no search results but we have FAQ answer, format it as a search result
        if not search_results and faq_answer:
            search_results = [{
                'title': faq_answer['question'],
                'link': '#',
                'snippet': faq_answer['answer'],
                'type': 'text',
                'image_url': None,
                'source': 'FAQ'
            }]
        
        # If still no results, try alternative search
        if not search_results:
            search_results = get_alternative_search_results(query)
            
            # If alternative search also fails, create a helpful response
            if not search_results or (len(search_results) == 1 and search_results[0]['title'] == 'No direct results found'):
                search_results = [{
                    'title': 'Information Available',
                    'link': '#',
                    'snippet': 'Here are some related topics that might help:\n\n' + 
                              '\n'.join([f"• {q}" for q in list(predefined_qa_pairs.keys())[:5]]),
                    'type': 'text',
                    'image_url': None,
                    'source': 'Suggested Topics'
                }]
        
        # Format response
        response = {
            'results': search_results,
            'count': len(search_results),
            'faq_answer': faq_answer
        }
        
        # Log the response for debugging
        print(f"Search response for query '{query}': {json.dumps(response, indent=2)}")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Search error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/search')
def search_google():
    """Direct Google search endpoint"""
    query = request.args.get('q', '')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        # Get search results
        results = google_search(query)
        
        # If no results, try FAQ
        if not results:
            faq_answer = get_faq_answer(query)
            if faq_answer:
                results = [{
                    'title': faq_answer['question'],
                    'snippet': faq_answer['answer'],
                    'type': 'text',
                    'link': '#',
                    'image_url': None
                }]
        
        # Format response
        response = {
            "results": results,
            "count": len(results)
        }
        
        # Log the response for debugging
        print(f"Google search response for query '{query}': {json.dumps(response, indent=2)}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/voice-input', methods=['POST'])
def voice_input_endpoint():
    """Voice input endpoint with error handling"""
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                return jsonify({'text': text})
            except sr.UnknownValueError:
                return jsonify({'error': 'Could not understand audio'}), 400
    except Exception as e:
        logger.error(f"Voice input error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-speech', methods=['POST'])
def stop_speech_endpoint():
    """Stop speech endpoint with error handling"""
    try:
        stop_speech()
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Stop speech error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Initialize models with retry mechanism
def initialize_models(max_retries=3):
    """Initialize AI models with retry mechanism"""
    for attempt in range(max_retries):
        try:
            # Initialize BERT
            global bert_tokenizer, bert_model
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertModel.from_pretrained('bert-base-uncased')
            bert_model.eval()
            
            # Initialize Sentence Transformer
            global sentence_transformer
            sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize Gemini
            global model
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-pro")
            
            logger.info("All models initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Model initialization attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            continue
    return False

# Initialize models
initialize_models()

# Enhanced error handling decorator
def handle_errors(func):
    """Decorator for enhanced error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return {"error": str(e)}
    return wrapper

@handle_errors
def get_faq_answer_with_multiple_strategies(question):
    """Get FAQ answer using multiple strategies with fallbacks"""
    strategies = [
        (get_faq_answer_with_bert, "BERT"),
        (get_faq_answer_with_fuzzy, "Fuzzy"),
        (get_faq_answer_with_gemini, "Gemini")
    ]
    
    for strategy_func, strategy_name in strategies:
        try:
            result = strategy_func(question)
            if result and result.get('answer'):
                return result
        except Exception as e:
            logger.error(f"{strategy_name} strategy failed: {str(e)}")
            continue
    
    return {"error": "No answer found using any strategy"}

@handle_errors
def get_faq_answer_with_bert(question):
    """Get FAQ answer using BERT with enhanced error handling"""
    if not bert_model or not sentence_transformer:
        return None
        
    try:
        best_match, bert_score = get_faq_answer_with_bert(question)
        if best_match and bert_score > 0.7:
            return {
                'answer': best_match[1],
                'question': best_match[0],
                'confidence': int(bert_score * 100),
                'source': 'BERT Match'
            }
    except Exception as e:
        logger.error(f"BERT matching failed: {str(e)}")
    return None

@handle_errors
def get_faq_answer_with_fuzzy(question):
    """Get FAQ answer using fuzzy matching"""
    try:
        best_match = None
        best_ratio = 0
        
        for q in predefined_qa_pairs.keys():
            ratio = fuzz.ratio(question.lower(), q.lower())
            if ratio > best_ratio and ratio > 80:
                best_ratio = ratio
                best_match = q
        
        if best_match:
            return {
                'answer': predefined_qa_pairs[best_match],
                'question': best_match,
                'confidence': best_ratio,
                'source': 'Fuzzy Match'
            }
    except Exception as e:
        logger.error(f"Fuzzy matching failed: {str(e)}")
    return None

@handle_errors
def get_faq_answer_with_gemini(question):
    """Get FAQ answer using Gemini AI"""
    try:
        if not model:
            return None
            
        prompt = f"""As an AI assistant for HITS College, please provide a helpful response to this query: {question}
        Keep the response concise, accurate, and relevant to HITS College information."""
        
        response = model.generate_content(prompt)
        if response and response.text:
            return {
                'answer': response.text,
                'question': question,
                'source': 'AI Generated',
                'confidence': 70
            }
    except Exception as e:
        logger.error(f"Gemini AI failed: {str(e)}")
    return None

@handle_errors
def get_faq_answer(query):
    """Get FAQ answer using simple matching"""
    try:
        # Convert query to lowercase for better matching
        query = query.lower()
        
        # First try exact match
        if query in predefined_qa_pairs:
            return {
                'question': query,
                'answer': predefined_qa_pairs[query],
                'source': 'Exact Match'
            }
        
        # Then try partial match
        for q, a in predefined_qa_pairs.items():
            if query in q or q in query:
                return {
                    'question': q,
                    'answer': a,
                    'source': 'Partial Match'
                }
        
        # Finally try fuzzy matching
        best_match = None
        best_score = 0
        
        for q in predefined_qa_pairs.keys():
            score = fuzz.ratio(query, q)
            if score > best_score and score > 70:
                best_score = score
                best_match = q
        
        if best_match:
            return {
                'question': best_match,
                'answer': predefined_qa_pairs[best_match],
                'source': 'Fuzzy Match'
            }
        
        return None
        
    except Exception as e:
        print(f"FAQ answer error: {str(e)}")
        return None

@app.route('/api/faq', methods=['GET'])
def get_faq():
    """Get FAQ questions with error handling"""
    try:
        questions = list(predefined_qa_pairs.keys())
        questions.sort()
        return jsonify({'questions': questions})
    except Exception as e:
        print(f"FAQ error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/faq/answer', methods=['POST'])
def get_faq_answer_endpoint():
    """Enhanced FAQ answer endpoint with multiple strategies"""
    try:
        data = request.get_json()
        question = data.get('question', '').lower().strip()
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # First try exact match
        if question in predefined_qa_pairs:
            return jsonify({
                'question': question,
                'answer': predefined_qa_pairs[question],
                'source': 'Predefined QA Pairs',
                'confidence': 100
            })

        # Then try fuzzy matching
        best_match = find_best_match(question, predefined_qa_pairs)
        if best_match:
            return jsonify({
                'question': best_match,
                'answer': predefined_qa_pairs[best_match],
                'source': 'Predefined QA Pairs (Fuzzy Match)',
                'confidence': 85
            })

        # Finally, try semantic search
        semantic_answer = semantic_search(question, predefined_qa_pairs)
        if semantic_answer:
            return jsonify({
                'question': question,
                'answer': semantic_answer,
                'source': 'Predefined QA Pairs (Semantic Match)',
                'confidence': 75
            })

        return jsonify({
            'error': 'No answer found for this question',
            'question': question
        }), 404

    except Exception as e:
        print(f"FAQ answer error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Initialize the application
def init_app():
    """Initialize the application with required setup"""
    try:
        validate_api_keys()
        load_qa_pairs()
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Application initialization failed: {str(e)}")
        raise

# Initialize on startup
init_app()

if __name__ == '__main__':
    app.run(debug=True)