import os
import re
import fitz
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import streamlit as st
from io import BytesIO
import faiss
import pickle
import math


nltk.download('punkt_tab')
nltk.download('stopwords')


class DocumentProcessor:
    def __init__(self, output_dir="processed_documents"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        

    def extract_text_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    

    def process_document(self, pdf_path, doc_id):
        text = self.extract_text_from_pdf(pdf_path)
        
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        output_path = os.path.join(self.output_dir, f"{doc_id}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
            
        return text
    

    def process_directory(self, directory_path):
        document_contents = {}
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                doc_id = os.path.splitext(filename)[0]
                file_path = os.path.join(directory_path, filename)
                text = self.process_document(file_path, doc_id)
                document_contents[doc_id] = text
                
        return document_contents



class TextChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    

    def chunk_text(self, text, doc_id):
        section_pattern = re.compile(r'(?m)^((?:[A-Z][A-Za-z\s]+|\d+\.\s+[A-Za-z\s]+):?$)')
        sections = section_pattern.split(text)
        
        chunks = []
        current_text = ""
        current_section = "Introduction"
        
        for i in range(len(sections)):
            part = sections[i].strip()
            if i % 2 == 1:
                current_section = part
                continue
                
            if not part:
                continue
                
            if len(part) > self.chunk_size:
                sentences = sent_tokenize(part)
                current_chunk = []
                current_size = 0
                
                for sentence in sentences:
                    sentence_size = len(sentence)
                    
                    if current_size + sentence_size > self.chunk_size and current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append({
                            'doc_id': doc_id,
                            'section': current_section,
                            'text': chunk_text,
                            'size': current_size
                        })
                        
                        overlap_size = 0
                        overlap_chunk = []
                        
                        while overlap_size < self.chunk_overlap and current_chunk:
                            sent = current_chunk.pop(0)
                            overlap_chunk.append(sent)
                            overlap_size += len(sent)
                        
                        current_chunk = overlap_chunk
                        current_size = overlap_size
                    
                    current_chunk.append(sentence)
                    current_size += sentence_size
                
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'doc_id': doc_id,
                        'section': current_section,
                        'text': chunk_text,
                        'size': current_size
                    })
            else:
                chunks.append({
                    'doc_id': doc_id,
                    'section': current_section,
                    'text': part,
                    'size': len(part)
                })
        
        return chunks



class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = []
        self.chunks = []
        self.index = None
        self.dimension = self.model.get_sentence_embedding_dimension()
    

    def add_chunks(self, chunks):
        for chunk in chunks:
            embedding = self.model.encode(chunk['text'])
            self.embeddings.append(embedding)
            self.chunks.append(chunk)
    

    def build_index(self):
        if not self.embeddings:
            return
            
        embeddings_array = np.array(self.embeddings).astype('float32')
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_array)
    

    def search(self, query, k=5):
        query_vector = self.model.encode([query])[0].reshape(1, -1).astype('float32')
        
        if self.index is None:
            raise ValueError("Index has not been built yet")
            
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:   
                results.append({
                    'chunk': self.chunks[idx],
                    'score': float(1.0 / (1.0 + distances[0][i]))
                })
                
        return results
    

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.embeddings
            }, f)
    

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.embeddings = data['embeddings']
        self.build_index()



class BM25Retriever:   
    def __init__(self):
        self.chunks = []
        self.doc_freqs = {}
        self.term_freqs = []
        self.corpus_size = 0
        self.avg_doc_len = 0
        self.doc_lens = []
        self.k1 = 1.5
        self.b = 0.75
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        self.stop_words = set(stopwords.words('english'))
        

    def add_chunks(self, chunks):
        for chunk in chunks:
            self._add_document(chunk['text'], chunk)


    def _add_document(self, text, chunk_info):
        terms = [t.lower() for t in self.tokenizer.tokenize(text) 
                if t.lower() not in self.stop_words and len(t) > 1]
        
        doc_len = len(terms)
        self.doc_lens.append(doc_len)
        
        term_freqs = {}
        for term in terms:
            if term not in term_freqs:
                term_freqs[term] = 0
            term_freqs[term] += 1
            
            if term not in self.doc_freqs:
                self.doc_freqs[term] = 0
            self.doc_freqs[term] += 1
        
        self.term_freqs.append(term_freqs)
        self.chunks.append(chunk_info)
        
        self.corpus_size += 1
        self.avg_doc_len = sum(self.doc_lens) / self.corpus_size
    

    def search(self, query, k=5):
        if not self.chunks:
            return []
            
        query_terms = [t.lower() for t in self.tokenizer.tokenize(query) 
                    if t.lower() not in self.stop_words and len(t) > 1]
        
        scores = []
        for doc_id in range(self.corpus_size):
            score = self._bm25_score(query_terms, doc_id)
            scores.append((doc_id, score))
        
        raw_scores = [score for _, score in scores]
        if raw_scores and max(raw_scores) > 0:
            min_score = min(raw_scores)
            max_score = max(raw_scores)
            
            if max_score > min_score:
                scores = [(doc_id, (score - min_score) / (max_score - min_score)) 
                        for doc_id, score in scores]
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in scores[:k]:
            if score > 0:
                results.append({
                    'chunk': self.chunks[doc_id],
                    'score': score
                })
        
        return results
    

    def _bm25_score(self, query_terms, doc_id):
        score = 0
        doc_len = self.doc_lens[doc_id]
        
        for term in query_terms:
            if term not in self.doc_freqs:
                continue
                
            if term not in self.term_freqs[doc_id]:
                continue
                
            tf = self.term_freqs[doc_id][term]
            
            numerator = self.corpus_size - self.doc_freqs[term] + 0.5
            denominator = self.doc_freqs[term] + 0.5
            
            if numerator <= 0 or denominator <= 0:
                continue
                
            log_val = numerator / denominator
            if log_val <= 0:
                continue
                
            idf = math.log(log_val)
            
            idf = max(0, idf)
            
            numerator = idf * tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(1, self.avg_doc_len))
            
            if denominator == 0:
                continue
                
            score += numerator / denominator
            
        return score



class EnhancedHybridSearch:   
    def __init__(self, vector_store, bm25_retriever, vector_weight=0.6):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.vector_weight = vector_weight
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        self.stop_words = set(stopwords.words('english'))
        

    def search(self, query, k=5):
        query_terms = [t.lower() for t in self.tokenizer.tokenize(query) 
                      if t.lower() not in self.stop_words and len(t) > 1]
        
        vector_results = self.vector_store.search(query, k=k*2)
        
        bm25_results = self.bm25_retriever.search(query, k=k*2)
        
        combined_results = {}
        
        for result in vector_results:
            chunk = result['chunk']
            chunk_id = f"{chunk['doc_id']}_{hash(chunk['text']) % 100000}"
            
            if chunk_id not in combined_results:
                combined_results[chunk_id] = {
                    'chunk': chunk,
                    'score': 0,
                    'term_match_score': self._calculate_term_match(query_terms, chunk['text']),
                    'section_relevance': self._section_relevance(query, chunk.get('section', ''))
                }
            
            combined_results[chunk_id]['score'] += result['score'] * self.vector_weight
        
        for result in bm25_results:
            chunk = result['chunk']
            chunk_id = f"{chunk['doc_id']}_{hash(chunk['text']) % 100000}"
            
            if chunk_id not in combined_results:
                combined_results[chunk_id] = {
                    'chunk': chunk,
                    'score': 0,
                    'term_match_score': self._calculate_term_match(query_terms, chunk['text']),
                    'section_relevance': self._section_relevance(query, chunk.get('section', ''))
                }
            
            combined_results[chunk_id]['score'] += result['score'] * (1 - self.vector_weight)
        
        for result in combined_results.values():
            result['score'] *= (1 + result['term_match_score'] * 0.5)
            result['score'] *= (1 + result['section_relevance'] * 0.3)
        
        results = list(combined_results.values())
        results.sort(key=lambda x: x['score'], reverse=True)
        
        final_results = self._diversify_results(results, k)
        
        return [{'doc_id': r['chunk']['doc_id'], 'chunks': [r['chunk']], 'score': r['score']} 
                for r in final_results]
    

    def _calculate_term_match(self, query_terms, text):
        if not query_terms:
            return 0
            
        text_lower = text.lower()
        matches = sum(1 for term in query_terms if term in text_lower)
        return matches / len(query_terms)
    

    def _section_relevance(self, query, section):
        if not section:
            return 0
            
        query_emb = self.vector_store.model.encode([query])[0]
        section_emb = self.vector_store.model.encode([section])[0]
        
        similarity = np.dot(query_emb, section_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(section_emb))
        return max(0, similarity)
    

    def _diversify_results(self, results, k):
        if len(results) <= 1:
            return results
            
        selected = [results[0]]
        remaining = results[1:]
        
        while len(selected) < k and remaining:
            max_diversity_score = -1
            best_idx = 0
            
            for i, result in enumerate(remaining):
                avg_similarity = 0
                for sel in selected:
                    text1 = result['chunk']['text'].lower()
                    text2 = sel['chunk']['text'].lower()
                    
                    terms1 = set(self.tokenizer.tokenize(text1))
                    terms2 = set(self.tokenizer.tokenize(text2))
                    
                    if not terms1 or not terms2:
                        continue
                        
                    intersection = len(terms1.intersection(terms2))
                    union = len(terms1.union(terms2))
                    
                    similarity = intersection / union if union > 0 else 0
                    avg_similarity += similarity
                
                avg_similarity /= len(selected) if len(selected) > 0 else 1
                
                diversity_score = result['score'] * (1 - avg_similarity * 0.7)
                
                if diversity_score > max_diversity_score:
                    max_diversity_score = diversity_score
                    best_idx = i
            
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)
        
        return selected



class ImprovedResponseGenerator:    
    def generate_response(self, query, search_results):
        if not search_results:
            return "I couldn't find relevant information for your query. Please try rephrasing your question or being more specific about the tax law topic you're interested in."
        
        query_analysis = self._analyze_query(query)
        
        consolidated_info = self._extract_key_information(query, search_results, query_analysis)
        
        return self._format_response(query, consolidated_info, search_results)
    

    def _analyze_query(self, query):
        question_words = ["what", "how", "when", "where", "why", "who", "which", "can", "do", "is", "are"]
        has_question = any(query.lower().startswith(word) for word in question_words) or "?" in query
        
        command_words = ["explain", "describe", "compare", "list", "define", "summarize", "analyze"]
        command_intent = next((word for word in command_words if word in query.lower()), None)
        
        tokens = nltk.word_tokenize(query.lower())
        stop_words = set(stopwords.words('english') + question_words + command_words)
        key_terms = [word for word in tokens if word.isalnum() and word not in stop_words and len(word) > 2]
        
        tax_terms = ["tax", "deduction", "credit", "liability", "income", "corporate", "vat", "withholding", 
                    "exemption", "filing", "return", "audit", "treaty", "jurisdiction"]
        tax_specific_terms = [term for term in key_terms if any(tax_term in term for tax_term in tax_terms)]
        
        return {
            "is_question": has_question,
            "command_intent": command_intent,
            "key_terms": key_terms,
            "tax_specific_terms": tax_specific_terms
        }
    

    def _extract_key_information(self, query, search_results, query_analysis):
        """Extract and organize key information from search results"""
        info = {
            "key_points": [],
            "definitions": [],
            "examples": [],
            "citations": [],
            "related_concepts": set()
        }
        
        for result in search_results:
            for chunk in result.get('chunks', []):
                text = chunk['text']
                doc_id = chunk['doc_id']
                section = chunk.get('section', '')
                
                definition_patterns = [
                    r'([A-Z][a-zA-Z\s]+) (?:is defined as|means|refers to|is) ([^\.]+)',
                    r'([A-Z][a-zA-Z\s]+):\s*([^\.]+)'
                ]
                
                for pattern in definition_patterns:
                    for match in re.finditer(pattern, text):
                        term, definition = match.groups()
                        if any(key_term in term.lower() for key_term in query_analysis["key_terms"]):
                            info["definitions"].append((term.strip(), definition.strip(), doc_id))
                
                sentences = sent_tokenize(text)
                for sentence in sentences:
                    if any(term in sentence.lower() for term in query_analysis["key_terms"]):
                        if len(sentence) > 30 and not sentence.startswith("Note that"):
                            info["key_points"].append((sentence, doc_id, section))
                
                example_indicators = ["for example", "e.g.", "such as", "for instance", "to illustrate"]
                for sentence in sentences:
                    if any(indicator in sentence.lower() for indicator in example_indicators):
                        if any(term in sentence.lower() for term in query_analysis["key_terms"]):
                            info["examples"].append((sentence, doc_id))
                
                citation_patterns = [
                    r'(?:under|according to|as per|pursuant to) ([^\.]+)',
                    r'(?:Section|Article|Regulation|Code) (\d+(?:\.\d+)*[a-z]?)'
                ]
                
                for pattern in citation_patterns:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        citation = match.group(1).strip()
                        info["citations"].append((citation, doc_id))
                
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    if any(term in sentence_lower for term in query_analysis["key_terms"]):
                        for term in nltk.word_tokenize(sentence):
                            if (term.lower() not in query_analysis["key_terms"] and
                                term.lower() not in stopwords.words('english') and
                                term.isalpha() and len(term) > 3):
                                info["related_concepts"].add(term)
        
        info["key_points"] = sorted(set(info["key_points"]), key=lambda x: x[0])[:7]
        info["definitions"] = sorted(set(info["definitions"]), key=lambda x: x[0])[:3]
        info["examples"] = sorted(set(info["examples"]), key=lambda x: x[0])[:2]
        info["citations"] = sorted(set(info["citations"]), key=lambda x: x[0])[:5]
        info["related_concepts"] = sorted(list(info["related_concepts"]))[:5]
        
        return info
    

    def _format_response(self, query, info, search_results):
        response = f"Based on the tax law documentation, here's what I found about '{query}':\n\n"
        
        if info["definitions"]:
            response += "**Key Definitions:**\n"
            for term, definition, doc_id in info["definitions"]:
                response += f"• **{term}**: {definition} (Source: Document {doc_id})\n"
            response += "\n"
        
        if info["key_points"]:
            response += "**Key Information:**\n"
            for point, doc_id, section in info["key_points"]:
                section_info = f" ({section})" if section else ""
                response += f"• {point} (Source: Document {doc_id}{section_info})\n\n"
            response += "\n"
        
        if info["examples"]:
            response += "**Examples:**\n"
            for example, doc_id in info["examples"]:
                response += f"• {example} (Source: Document {doc_id})\n"
            response += "\n"
        
        if info["citations"]:
            response += "**Relevant Citations:**\n"
            unique_citations = set()
            for citation, doc_id in info["citations"]:
                if citation not in unique_citations:
                    unique_citations.add(citation)
                    response += f"• {citation} (Referenced in Document {doc_id})\n"
            response += "\n"
        
        doc_ids = set()
        for result in search_results:
            doc_ids.add(result["doc_id"])
        
        response += "**Source Documents:** " + ", ".join(sorted(doc_ids)) + "\n\n"
        
        response += "*Note: This information is based on the available tax law documents. For definitive guidance, please consult with a qualified tax professional.*"
        
        return response



def create_ui():
    st.set_page_config(page_title="International Tax Law Assistant", layout="wide")
    
    st.title("LegisTax: International Tax Law Research Assistant")
    
    models_exist = os.path.exists("models/vector_store.pkl") and os.path.exists("models/search_engine.pkl")
    
    if models_exist and 'search_engine' not in st.session_state:
        with st.spinner("Loading existing document index..."):
            try:
                vector_store = VectorStore()
                vector_store.load("models/vector_store.pkl")
                
                with open("models/search_engine.pkl", "rb") as f:
                    search_engine = pickle.load(f)
                    
                st.session_state.search_engine = search_engine
                st.session_state.response_generator = ImprovedResponseGenerator()
                
                st.success(f"Loaded existing document index with {len(vector_store.chunks)} document chunks.")
            except Exception as e:
                st.error(f"Error loading existing models: {str(e)}")
    with st.sidebar:
        st.header("Document Management")

        if models_exist:
            st.info(f"Existing document index found.")
            if st.button("Clear existing index"):
                if os.path.exists("models/vector_store.pkl"):
                    os.remove("models/vector_store.pkl")
                if os.path.exists("models/search_engine.pkl"):
                    os.remove("models/search_engine.pkl")
                if 'search_engine' in st.session_state:
                    del st.session_state.search_engine
                if 'response_generator' in st.session_state:
                    del st.session_state.response_generator
                st.success("Index cleared. Please refresh the page.")
        
        uploaded_files = st.file_uploader("Upload tax law documents", accept_multiple_files=True, type="pdf")
        
        
        if st.button("Process Uploaded Documents"):
            if uploaded_files:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                processor = DocumentProcessor()
                chunker = TextChunker(chunk_size=800, chunk_overlap=200)
                vector_store = VectorStore()
                bm25_retriever = BM25Retriever()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    bytes_data = uploaded_file.getvalue()
                    doc_id = uploaded_file.name.split('.')[0]
                    temp_path = f"temp_{doc_id}.pdf"
                    
                    with open(temp_path, "wb") as f:
                        f.write(bytes_data)
                    
                    text = processor.process_document(temp_path, doc_id)
                    
                    chunks = chunker.chunk_text(text, doc_id)
                    
                    vector_store.add_chunks(chunks)
                    
                    bm25_retriever.add_chunks(chunks)
                    
                    os.remove(temp_path)
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("Building search index...")
                vector_store.build_index()
                
                search_engine = EnhancedHybridSearch(vector_store, bm25_retriever)
                
                if not os.path.exists("models"):
                    os.makedirs("models")
                
                vector_store.save("models/vector_store.pkl")
                
                with open("models/search_engine.pkl", "wb") as f:
                    pickle.dump(search_engine, f)
                
                status_text.text("Processing complete!")
                
                st.session_state.search_engine = search_engine
                st.session_state.response_generator = ImprovedResponseGenerator()
                
                st.rerun()
    
    st.header("Query Tax Law Documents")
    
    if 'search_engine' not in st.session_state:
        st.info("Please upload and process documents using the sidebar to get started.")
    else:
        query = st.text_input("Enter your tax law query:")
        
        if query:
            search_engine = st.session_state.search_engine
            response_generator = st.session_state.response_generator
            
            with st.spinner("Searching..."):
                search_results = search_engine.search(query, k=7)
                
                response = response_generator.generate_response(query, search_results)
                
                st.subheader("Response:")
                st.write(response)
                
                st.subheader("Detailed Sources:")
                
                for i, result in enumerate(search_results):
                    with st.expander(f"Source {i+1}: Document {result['doc_id']} (Score: {result['score']:.2f})"):
                        for chunk in result.get('chunks', []):
                            st.text_area(f"Extract from {result['doc_id']}", chunk['text'], height=200)



if __name__ == "__main__":
    create_ui()