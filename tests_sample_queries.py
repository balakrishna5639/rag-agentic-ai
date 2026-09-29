import requests
import time
from colorama import Fore, Style, init

# Initialize colorama for colored terminal output
init(autoreset=True)

API_URL = "http://127.0.0.1:8000/chat"

# Benchmark testing queries based on assignment requirements
TEST_QUERIES = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "What is the capital of France?" # Out-of-Scope check
]

def run_tests():
    print(f"{Fore.CYAN}{Style.BRIGHT}==========================================")
    print(f"{Fore.CYAN}{Style.BRIGHT}   RAG Chatbot Benchmarking & QA Script   ")
    print(f"{Fore.CYAN}{Style.BRIGHT}==========================================\n")
    
    passed_tests = 0
    total_tests = len(TEST_QUERIES)

    for idx, query in enumerate(TEST_QUERIES, 1):
        print(f"{Fore.YELLOW}[Test {idx}/{total_tests}] Query: {query}")
        
        payload = {"query": query}
        
        try:
            start_time = time.time()
            response = requests.post(API_URL, json=payload)
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"{Fore.GREEN}Status: SUCCESS ({elapsed_time:.2f}s)")
                print(f"{Fore.WHITE}Answer: {data.get('final_answer')}")
                print(f"{Fore.MAGENTA}Confidence Score: {data.get('confidence_score')}")
                print(f"{Fore.BLUE}Retrieved Context Chunks: {len(data.get('retrieved_context_chunks', []))}\n")
                passed_tests += 1
            else:
                print(f"{Fore.RED}Status: FAILED (HTTP {response.status_code})")
                print(f"{Fore.RED}Error: {response.text}\n")
                
        except requests.exceptions.ConnectionError:
            print(f"{Fore.RED}Status: FAILED (Connection Error)")
            print(f"{Fore.RED}Is the FastAPI server running on {API_URL}?\n")
            break
            
    print(f"{Fore.CYAN}{Style.BRIGHT}==========================================")
    print(f"{Fore.CYAN}{Style.BRIGHT}   Testing Summary: {passed_tests}/{total_tests} Tests Passed")
    print(f"{Fore.CYAN}{Style.BRIGHT}==========================================")

if __name__ == "__main__":
    run_tests()
