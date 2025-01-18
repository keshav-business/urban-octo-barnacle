from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import json

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage class
class QuizStorage:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.user_sessions: Dict[str, Dict] = {}

quiz_storage = QuizStorage()

# Pydantic models
class QuizResponse(BaseModel):
    question_id: int
    answer: str

class QuizResult(BaseModel):
    total_score: float
    max_score: float
    feedback: List[Dict[str, str]]
    improvement_areas: List[str]

# Initialize PDFs at startup
@app.on_event("startup")
async def startup_event():
    pdf_dir = "datamn"  # Directory containing your PDFs
    try:
        documents = []
        for pdf_file in os.listdir(pdf_dir):
            if pdf_file.endswith('.pdf'):
                loader = PyPDFLoader(os.path.join(pdf_dir, pdf_file))
                documents.extend(loader.load())
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        quiz_storage.vector_store = FAISS.from_documents(splits, embeddings)
        
        # Create QA chain
        llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
        quiz_storage.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=quiz_storage.vector_store.as_retriever(),
        )
    except Exception as e:
        print(f"Error initializing PDFs: {str(e)}")
        raise

def generate_unique_questions() -> List[str]:
    """Generate unique questions from PDF content."""
    prompt = """Based on the content in the PDFs, generate 5 unique and different questions that test understanding 
    of key concepts. Make sure questions are detailed and require explanatory answers. 
    Return the questions as a JSON array of strings. Each question should be different from others and cover different topics from the PDF.
    The questions should encourage detailed explanations rather than simple facts."""
    
    try:
        response = quiz_storage.qa_chain.run(prompt)
        # Clean up the response and parse JSON
        response = response.replace("'", '"')
        # Extract JSON array if embedded in text
        if '[' in response and ']' in response:
            response = response[response.find('['):response.rfind(']')+1]
        questions = json.loads(response)
        return questions[:5]  # Ensure we get exactly 5 questions
    except Exception as e:
        print(f"Error generating questions: {e}")
        # Fallback questions in case of error
        return [
            "Please explain a key concept from the provided materials.",
            "What are the main points discussed in the documents?",
            "Describe one of the important topics covered.",
            "What are the implications of the concepts discussed?",
            "How would you apply the knowledge from these materials?"
        ]

@app.post("/start-quiz/{user_id}")
async def start_quiz(user_id: str):
    if not quiz_storage.qa_chain:
        raise HTTPException(status_code=500, detail="Quiz system not properly initialized")
    
    # Generate unique questions for this session
    questions = generate_unique_questions()
    
    quiz_storage.user_sessions[user_id] = {
        "current_question": 0,
        "answers": [],
        "scores": [],
        "feedback": [],
        "questions": questions  # Store questions for this session
    }
    
    return get_next_question(user_id)

async def answer_question(user_id: str, response: QuizResponse):
    if user_id not in quiz_storage.user_sessions:
        raise HTTPException(status_code=404, detail="Quiz session not found")
    
    session = quiz_storage.user_sessions[user_id]
    
    if session["current_question"] >= 5:
        raise HTTPException(status_code=400, detail="Quiz already completed")
    
    current_question = session["questions"][session["current_question"]]
    
    # Check if question was skipped
    if response.answer == "SKIPPED":
        # Get ideal answer for skipped question
        ideal_answer_prompt = f"""
        Based on the content in the PDFs, provide a detailed ideal answer for this question:
        Question: {current_question}
        
        Format your response as a clear, concise explanation that could serve as a model answer.
        """
        
        ideal_answer = quiz_storage.qa_chain.run(ideal_answer_prompt)
        
        # Record skipped question results
        session["answers"].append("SKIPPED")
        session["scores"].append(0)  # Zero score for skipped questions
        session["feedback"].append({
            "question": current_question,
            "feedback": "Question was skipped",
            "improvement": "Study the provided ideal answer",
            "ideal_answer": ideal_answer
        })
        session["current_question"] += 1
        
        if session["current_question"] >= 5:
            return calculate_results(user_id)
        else:
            return get_next_question(user_id)

def get_next_question(user_id: str) -> Dict:
    session = quiz_storage.user_sessions[user_id]
    return {
        "question_number": session["current_question"] + 1,
        "question_text": session["questions"][session["current_question"]]
    }

def calculate_results(user_id: str) -> QuizResult:
    session = quiz_storage.user_sessions[user_id]
    
    total_score = sum(session["scores"])
    
    return QuizResult(
        total_score=total_score,
        max_score=5.0,
        feedback=[{
            "question": fb["question"],
            "feedback": fb["feedback"]
        } for fb in session["feedback"]],
        improvement_areas=[fb["improvement"] for fb in session["feedback"]]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)